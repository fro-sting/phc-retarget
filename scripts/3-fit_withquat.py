import os
import torch
import torch.nn as nn
import hydra
import numpy as np
import glob
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
import multiprocessing as mp
import joblib

from smplx import SMPL
from scipy.spatial.transform import Rotation as sRot, Slerp

os.environ["OMP_NUM_THREADS"] = "1"

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")

SMPL_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

def parse_joint_limits_from_mjcf(root):
    limits = {}
    for j in root.findall(".//joint"):
        jtype = j.get("type", "hinge")
        if jtype not in ("hinge", "slide"):
            continue
        name = j.get("name")
        rng = j.get("range")
        if name is None or rng is None:
            continue
        low, high = map(float, rng.split())
        limits[name] = (low, high)
    return limits


def build_chain(cfg) -> pk.Chain:
    mjcf_path = cfg.asset.assetFileName

    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # remove the free joint of the base link
    root_name = cfg.get("root_name", "pelvis")
    root_body = root.find(f".//body[@name='{root_name}']")
    root_joint = root.find(".//joint[@type='free']")
    root_body.remove(root_joint)
    root_body.set("pos", "0 0 0")

    for extend_config in cfg.extend_config:
        parent = root.find(f".//body[@name='{extend_config.parent_name}']")
        if parent is None:
            raise ValueError(f"Parent body {extend_config.parent_name} not found in MJCF")
        
        pos = extend_config.pos
        rot = extend_config.rot
        # create and insert a dummy body with a fixed joint
        body = ET.Element("body", name=extend_config.joint_name)
        body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        body.set("quat", f"{rot[0]} {rot[1]} {rot[2]} {rot[3]}")
        inertial = ET.Element("inertial", pos="0 0 0", quat="0 0 0 1", mass="0.1", diaginertia="0.1 0.1 0.1")
        body.append(inertial)
        parent.append(body)

    cwd = os.getcwd()
    os.chdir(os.path.dirname(mjcf_path))
    chain = pk.build_chain_from_mjcf(ET.tostring(root, method="xml"), body=root_name)
    os.chdir(cwd)

    joint_limits = parse_joint_limits_from_mjcf(root)
    return chain, joint_limits


def lerp(x, xp, fp):
    return np.stack([np.interp(x, xp, fp[:, i]) for i in range(fp.shape[1])], axis=1)


def slerp(x, xp, fp):
    s = Slerp(xp, sRot.from_rotvec(fp))
    return s(x).as_rotvec()

def rodrigues(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Converts axis-angle representation to rotation matrix using Rodrigues' rotation formula.
    axis_angle: [N, 3]
    Returns: [N, 3, 3]
    """
    N = axis_angle.shape[0]
    device = axis_angle.device
    dtype = axis_angle.dtype
    
    theta = torch.norm(axis_angle, p=2, dim=1, keepdim=True) 

    I = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)

    mask = (theta > 1e-8).squeeze() # [N]

    R = I.clone()
    
    if not mask.any():
        return R

    aa_non_zero = axis_angle[mask]
    theta_non_zero = theta[mask].unsqueeze(-1) # [N_non_zero, 1, 1]

    k = aa_non_zero / theta_non_zero.squeeze(-1) # [N_non_zero, 3]

    K = torch.zeros((k.shape[0], 3, 3), device=device, dtype=dtype)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    
    sin_theta = torch.sin(theta_non_zero)
    cos_theta = torch.cos(theta_non_zero)
    # R = I + sin(theta)K + (1 - cos(theta))K^2
    R_non_zero = I[mask] + K * sin_theta + torch.matmul(K, K) * (1.0 - cos_theta)

    R[mask] = R_non_zero
    
    return R
def rotation_matrix_distance(R1, R2):
    """
    Calculates the geodesic distance (angle) between two batches of rotation matrices.
    R1, R2: [..., 3, 3]
    Returns: [...,]
    """
    R_rel = torch.matmul(R1.transpose(-1, -2), R2)
    
    # Trace
    tr = torch.einsum('...ii->...', R_rel)
    # cos(theta) = (Trace(R) - 1) / 2
    cos_theta = (tr - 1) / 2

    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    return torch.acos(cos_theta)


def fit_motion(cfg, motion_path: str, fitted_shape: torch.Tensor):
    
    with open(motion_path, "rb") as f:
        motion = dict(np.load(f, allow_pickle=True))
    
    if "mocap_framerate" not in motion and "mocap_frame_rate" not in motion:
        print(f"Skipping motion file (no mocap_framerate or mocap_frame_rate): {motion_path}")
        return None
    
    fps = int(motion["mocap_framerate"].item()) if "mocap_framerate" in motion else int(motion["mocap_frame_rate"].item())
    T = motion["poses"].shape[0]
    motion["poses"] = motion["poses"][:, :66].reshape(T, 22, 3)
    if fps != int(cfg.target_fps):
        end_t =  motion["poses"].shape[0] / fps
        xp = np.arange(T) / fps
        x = np.arange(0, end_t, 1 / int(cfg.target_fps))
        if x[-1] > xp[-1]:
            x = x[:-1]
        motion["poses"] = np.stack([
            slerp(x, xp, motion["poses"][:, i])
            for i in range(22)
        ], axis=1)
        motion["trans"] = lerp(x, xp, motion["trans"])
    
    print(f"Retargeting motion at {motion_path} from {fps} to {cfg.target_fps}")

    chain, joint_limits = build_chain(cfg)
    chain.forward_kinematics(torch.zeros(1, chain.n_joints))

    joint_names = chain.get_joint_parameter_names()
    assert len(joint_names) == len(joint_limits), f"Number of joints in chain ({len(joint_names)}) does not match number of joint limits ({len(joint_limits)})"

    low_list, high_list = [], []
    for joint_name, (joint_limit_key, joint_limit_val) in zip(joint_names, joint_limits.items()):
        assert joint_name == joint_limit_key, f"Joint name {joint_name} does not match joint name in joint limits ({joint_limit_key})"
        low_list.append(joint_limit_val[0])
        high_list.append(joint_limit_val[1])
    low = torch.as_tensor(low_list, dtype=torch.float32).unsqueeze(0)   # [1, J]
    high = torch.as_tensor(high_list, dtype=torch.float32).unsqueeze(0) # [1, J]

    T = motion["poses"].shape[0]
    body_pose = torch.as_tensor(motion["poses"][:, 1:], dtype=torch.float32)
    hand_pose = torch.zeros(T, 2, 3)
    data = {
        "body_pose": torch.cat([body_pose, hand_pose], dim=1),
        "global_orient": torch.as_tensor(motion["poses"][:, 0], dtype=torch.float32),
        "trans": torch.as_tensor(motion["trans"], dtype=torch.float32),
    }
    body_model = SMPL(model_path=os.path.join(DATA_PATH, "smpl"), gender="neutral")

    with torch.no_grad():
        result = body_model.forward(
            fitted_shape,
            body_pose=data["body_pose"].reshape(T, 69),
            global_orient=data["global_orient"],
            transl=data["trans"],
            return_full_pose=True
        )

    
   
    full_pose_aa = result.full_pose
    #print(f"Full pose shape: {full_pose_aa.shape}")
    T = full_pose_aa.shape[0]
    full_pose_aa_flat = full_pose_aa.view(-1, 3)
    smpl_local_mats_flat = rodrigues(full_pose_aa_flat)
    smpl_local_mats = smpl_local_mats_flat.view(T, 24, 3, 3)

    parents = body_model.parents
    
    #compute global matrices
    smpl_global_mats_w = torch.zeros_like(smpl_local_mats)
    smpl_global_mats_w[:, 0] = smpl_local_mats[:, 0]
    for i in range(1, 24):
        parent_idx = parents[i]
        smpl_global_mats_w[:, i] = torch.matmul(
            smpl_global_mats_w[:, parent_idx],
            smpl_local_mats[:, i]
        )

    # apply orientation offsets
    smpl_global_mats_modified = smpl_global_mats_w.clone()
    for joint_name, quat in cfg.quat_offset.items():
        joint_idx = SMPL_BONE_ORDER_NAMES.index(joint_name)
        
        # transform string to list
        if isinstance(quat, str):
            quat_str = quat.strip('[]')
            quat = [float(x.strip()) for x in quat_str.split(',')]
        
        # transform to rotation matrix
        rot = sRot.from_quat(quat, scalar_first=True)
        rot_mat = torch.as_tensor(rot.as_matrix(), dtype=torch.float32)
        #print(f"Applying quat offset to joint {joint_name} (index {joint_idx}): quat={quat}\nrot_mat=\n{rot_mat.cpu().numpy()}")

        smpl_global_mats_modified[:, joint_idx] = torch.matmul(
            smpl_global_mats_w[:, joint_idx],
            rot_mat
        )
        

    # which joints to match
    robot_body_names = []
    smpl_joint_idx = []
    for robot_body_name, smpl_joint_name in cfg.joint_matches:
        robot_body_names.append(robot_body_name)
        smpl_joint_idx.append(SMPL_BONE_ORDER_NAMES.index(smpl_joint_name))
    
    smpl_target_mats = smpl_global_mats_modified[:, smpl_joint_idx]
   
   #print(f"joint_idx={smpl_joint_idx}   robot_body_names={robot_body_names}")


    

    # since the betas are changed and so are the SMPL body morphology,
    # we need to make some corrections to avoid ground pentration
    ground_offset = result.vertices[:, :, 2].min()
    smpl_keypoints_w = result.joints[:, smpl_joint_idx] - ground_offset

    # again, convert between Y-up and Z-up
    robot_rot = sRot.from_rotvec(data["global_orient"]) * sRot.from_euler("xyz", [np.pi/2, 0., np.pi/2]).inv()
    robot_rotmat = torch.as_tensor(robot_rot.as_matrix(), dtype=torch.float32)

    robot_th = torch.nn.Parameter(torch.zeros(T, chain.n_joints))        
    robot_trans = torch.nn.Parameter(data["trans"].clone() - ground_offset)
    opt = torch.optim.Adam([robot_th, robot_trans], lr=0.02)

    indices = chain.get_all_frame_indices()
    
    def mat_rotate(rotmat, v):
        return (rotmat @ v.unsqueeze(-1)).squeeze(-1)

    for i in range(500):
        fk_output = chain.forward_kinematics(robot_th, indices) # in robot's root frame
        robot_keypoints_b = torch.stack([
            fk_output[name].get_matrix()[:, :3, 3]
            for name in robot_body_names
        ], dim=1)        
        # convert to world frame
        robot_keypoints_w = robot_trans.unsqueeze(1) + mat_rotate(robot_rotmat.unsqueeze(1), robot_keypoints_b)

        robot_orient_mats_b = torch.stack([
            fk_output[name].get_matrix()[:, :3, :3]
            for name in robot_body_names
        ], dim=1) # Shape: [T, N, 3, 3]
        robot_orient_mats_w = torch.matmul(robot_rotmat.unsqueeze(1), robot_orient_mats_b)
        orient_angle_error = rotation_matrix_distance(robot_orient_mats_w, smpl_target_mats)
        orientation_error = torch.mean(torch.square(orient_angle_error))

        joint_orient_errors = torch.mean(torch.square(orient_angle_error), dim=0)  # [N]


        omega = torch.gradient(robot_th, spacing=1/cfg.target_fps, dim=0)[0]    # [T, J]
        
        violate_low  = torch.relu(low  - robot_th)
        violate_high = torch.relu(robot_th - high)
        L_limit = (violate_low**2 + violate_high**2).mean()

        keypoints_pos_error = nn.functional.mse_loss(robot_keypoints_w, smpl_keypoints_w)
        joint_pos_reg = 1e-2 * torch.mean(torch.square(robot_th))
        joint_vel_reg = 1e-3 * torch.mean(torch.square(omega))
        orientation_reg = 1e-1 * orientation_error
        joint_limit_reg = 1e2 * L_limit

        loss = keypoints_pos_error + joint_pos_reg + joint_vel_reg + joint_limit_reg + orientation_reg
        #loss = joint_pos_reg + joint_vel_reg + joint_limit_reg + orientation_reg
        #loss = keypoints_pos_error + joint_pos_reg + joint_vel_reg + joint_limit_reg
        


        opt.zero_grad()
        loss.backward()
        opt.step()
        # if i % 10 == 0:
        #     print(f"iter {i}, loss {100 * loss.item():.3f}")

    with torch.no_grad():
        robot_keypoints_b = torch.stack([
            fk_output[name].get_matrix()[:, :3, 3]
            for name in chain.get_link_names()
        ], dim=1)        
        # convert to world frame
        robot_keypoints_w = robot_trans.unsqueeze(1) + mat_rotate(robot_rotmat.unsqueeze(1), robot_keypoints_b)

    split_len = len(cfg.motion_path.split("/"))
    motion_name = "0-" + "_".join(motion_path.split("/")[split_len:]).replace(".npz", "")
    data = {
        "fps": int(cfg.target_fps),
        "joint_pos": robot_th.data.numpy(),
        "root_pos_w": robot_trans.data.numpy(),
        "root_quat_w": robot_rot.as_quat(),
        "body_pos_w": robot_keypoints_w.data.numpy(),
        "body_pos_b": robot_keypoints_b.data.numpy(),
    }
    return motion_name, data


def _worker(args):
    cfg, path, betas = args
    return fit_motion(cfg, path, betas)

@hydra.main(version_base=None, config_path="../cfg", config_name="unitree_g1_fitting")
def main(cfg):
    if os.path.isdir(cfg.motion_path):
        motion_paths = glob.glob(os.path.join(cfg.motion_path, "**/*.npz"), recursive=True)
    else:
        motion_paths = [cfg.motion_path]

    motion_paths = [path for path in motion_paths]
    print(f"Found {len(motion_paths)} motion files under {cfg.motion_path}")

    path = os.path.join(os.path.dirname(__file__), f"{cfg.humanoid_type}_shape.npz")
    fitted_shape = torch.from_numpy(np.load(path)["betas"])


    from tqdm import tqdm
    all_data = {}
    with mp.get_context("spawn").Pool(
        processes=6,
        maxtasksperchild=1,
    ) as pool, tqdm(total=len(motion_paths)) as pbar:
        for result in pool.imap_unordered(
            _worker,
            [(cfg, p, fitted_shape) for p in motion_paths],
            chunksize=1,
        ):
            if result is not None:
                motion_name, data = result
                all_data[motion_name] = data
            pbar.update(1)

    os.makedirs(f"data/{cfg.humanoid_type}", exist_ok=True)
    joblib.dump(all_data, f"data/{cfg.humanoid_type}/{cfg.output_name}.pkl")

if __name__ == "__main__":
    main()