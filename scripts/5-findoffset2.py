import os
import torch
import torch.nn as nn
import hydra
import numpy as np
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
import open3d as o3d

from smplx import SMPL
from scipy.spatial.transform import Rotation as sRot

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")

SMPL_BONE_ORDER_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
    "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
    "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]

def build_chain(cfg):
    mjcf_path = cfg.asset.assetFileName
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    root_name = cfg.get("root_name", "pelvis")
    root_body = root.find(f".//body[@name='{root_name}']")
    root_joint = root.find(".//joint[@type='free']")
    if root_joint is not None:
        root_body.remove(root_joint)
    root_body.set("pos", "0 0 0")

    for extend_config in cfg.extend_config:
        parent = root.find(f".//body[@name='{extend_config.parent_name}']")
        if parent is None:
            raise ValueError(f"Parent body {extend_config.parent_name} not found")
        
        body = ET.Element("body", name=extend_config.joint_name)
        body.set("pos", f"{extend_config.pos[0]} {extend_config.pos[1]} {extend_config.pos[2]}")
        body.set("quat", f"{extend_config.rot[0]} {extend_config.rot[1]} {extend_config.rot[2]} {extend_config.rot[3]}")
        inertial = ET.Element("inertial", pos="0 0 0", quat="0 0 0 1", mass="0.1", diaginertia="0.1 0.1 0.1")
        body.append(inertial)
        parent.append(body)

    cwd = os.getcwd()
    os.chdir(os.path.dirname(mjcf_path))
    chain = pk.build_chain_from_mjcf(ET.tostring(root, method="xml"), body=root_name)
    os.chdir(cwd)
    
    return chain

def rodrigues(axis_angle: torch.Tensor) -> torch.Tensor:
    N = axis_angle.shape[0]
    device = axis_angle.device
    dtype = axis_angle.dtype
    
    theta = torch.norm(axis_angle, p=2, dim=1, keepdim=True)
    I = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)
    mask = (theta > 1e-8).squeeze()
    R = I.clone()
    
    if not mask.any():
        return R
        
    aa_non_zero = axis_angle[mask]
    theta_non_zero = theta[mask].unsqueeze(-1)
    k = aa_non_zero / theta_non_zero.squeeze(-1)
    
    K = torch.zeros((k.shape[0], 3, 3), device=device, dtype=dtype)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    
    sin_theta = torch.sin(theta_non_zero)
    cos_theta = torch.cos(theta_non_zero)
    
    R_non_zero = I[mask] + K * sin_theta + torch.matmul(K, K) * (1.0 - cos_theta)
    R[mask] = R_non_zero
    
    return R

def compute_offsets(cfg):
    """compute orientation offsets between SMPL and robot model in T-pose"""
    print("Loading robot chain...")
    chain = build_chain(cfg)
    
    print("Loading SMPL model...")
    path = os.path.join(os.path.dirname(__file__), f"{cfg.humanoid_type}_shape.npz")
    fitted_shape = torch.from_numpy(np.load(path)["betas"]).float()
    body_model = SMPL(model_path=os.path.join(DATA_PATH, "smpl"), gender="neutral")
    
    # ✅ 机器人T-pose: 所有关节角为0
    robot_th = torch.zeros(1, chain.n_joints, dtype=torch.float32)
    
    # get robot keypoints and orientations
    robot_body_names = []
    smpl_joint_idx = []
    for robot_body_name, smpl_joint_name in cfg.joint_matches:
        robot_body_names.append(robot_body_name)
        smpl_joint_idx.append(SMPL_BONE_ORDER_NAMES.index(smpl_joint_name))
    
    # ✅ 机器人FK (T-pose)
    indices = chain.get_all_frame_indices()
    fk_output = chain.forward_kinematics(robot_th, indices)
    
    # 获取机器人各关节的全局旋转矩阵
    robot_orient_mats = torch.stack([
        fk_output[name].get_matrix()[:, :3, :3]
        for name in robot_body_names
    ], dim=1)  # [1, N, 3, 3]
    
    print(f"Matching {len(robot_body_names)} joints")
    print("robot_orient_mats", robot_orient_mats[0])
    # ✅ SMPL T-pose: body_pose和global_orient都为0
    with torch.no_grad():
        result = body_model.forward(
            fitted_shape,
            body_pose=torch.zeros(1, 69),      # ✅ 全零
            global_orient=torch.zeros(1, 3),   # ✅ 全零
            transl=torch.zeros(1, 3),
            return_full_pose=True
        )
        
        # 计算SMPL的全局旋转矩阵
        full_pose_aa = result.full_pose  # [1, 24, 3]
        full_pose_aa_flat = full_pose_aa.view(-1, 3)
        smpl_local_mats_flat = rodrigues(full_pose_aa_flat)
        smpl_local_mats = smpl_local_mats_flat.view(1, 24, 3, 3)
        
        # 通过父节点层级计算全局旋转
        parents = body_model.parents
        smpl_global_mats = torch.zeros_like(smpl_local_mats)
        smpl_global_mats[:, 0] = smpl_local_mats[:, 0]
        for j in range(1, 24):
            parent_idx = parents[j]
            smpl_global_mats[:, j] = torch.matmul(
                smpl_global_mats[:, parent_idx],
                smpl_local_mats[:, j]
            )
        
        smpl_target_mats = smpl_global_mats[:, smpl_joint_idx]  # [1, N, 3, 3]
    print("smpl_target_mats", smpl_target_mats[0])
    # ✅ 计算offset: R_offset = R_smpl^T @ R_robot
    # 含义: R_robot = R_smpl @ R_offset
    # 即: 从SMPL旋转出发，再乘上offset，得到机器人旋转
    offsets = {}
    print("\nComputed offsets (T-pose to T-pose):")
    for idx, (robot_name, smpl_name) in enumerate(cfg.joint_matches):
        R_robot = robot_orient_mats[0, idx].numpy()  # [3, 3]
        R_smpl = smpl_target_mats[0, idx].numpy()    # [3, 3]
        
        # R_offset = R_smpl^T @ R_robot
        R_offset = R_smpl.T @ R_robot

        rot = sRot.from_matrix(R_offset)
        quat = rot.as_quat(scalar_first=True)  # (w, x, y, z)
        
        offsets[smpl_name] = quat.tolist()
        print(f"  {smpl_name}: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
        
        # ✅ 验证
        R_reconstructed = R_smpl @ R_offset
        error = np.linalg.norm(R_reconstructed - R_robot, 'fro')
        if error > 1e-5:
            print(f"    ⚠️ Reconstruction error: {error:.6f}")

    # 保存
    output_path = f"data/{cfg.humanoid_type}_offsets.npz"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **offsets)
    print(f"\nOffsets saved to {output_path}")

    print("\nYAML config format:")
    print("quat_offset:")
    for joint_name, quat in offsets.items():
        print(f"  {joint_name}: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
    
     # ✅ 可视化对比（可选）
    print("\nVisualizing T-pose comparison...")
    visualize_tpose_comparison(body_model, fitted_shape, robot_body_names, 
                               robot_orient_mats, smpl_target_mats, result, smpl_joint_idx)  # ✅ 传入索引

    return offsets


def visualize_tpose_comparison(body_model, fitted_shape, robot_body_names, 
                               robot_orient_mats, smpl_orient_mats, smpl_result, smpl_joint_idx):  # ✅ 添加参数
    """可视化T-pose下的关节方向对比"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="T-Pose Comparison")

    # 坐标系
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(frame)

    # SMPL mesh
    vertices = smpl_result.vertices[0].detach().cpu().numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(body_model.faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    vis.add_geometry(mesh)

    # 绘制关节方向（用箭头表示X轴）
    joints = smpl_result.joints[0].detach().cpu().numpy()
    
    def create_arrow(origin, direction, color):
        """创建箭头表示坐标轴"""
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01,
            cone_radius=0.02,
            cylinder_height=0.1,
            cone_height=0.03
        )
        # 旋转箭头对齐到direction
        R = rotation_matrix_from_vectors([0, 0, 1], direction)
        arrow.rotate(R, center=[0, 0, 0])
        arrow.translate(origin)
        arrow.paint_uniform_color(color)
        return arrow
    
    def rotation_matrix_from_vectors(v1, v2):
        """计算从v1到v2的旋转矩阵"""
        v1 = np.array(v1) / np.linalg.norm(v1)
        v2 = np.array(v2) / np.linalg.norm(v2)
        v = np.cross(v1, v2)
        s = np.linalg.norm(v)
        c = np.dot(v1, v2)
        
        if s < 1e-6:
            return np.eye(3)
        
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
        return R
    
    # ✅ 修复：使用传入的 smpl_joint_idx
    for idx, joint_idx in enumerate(smpl_joint_idx):
        origin = joints[joint_idx]
        
        # SMPL方向 (蓝色 - X轴)
        R_smpl = smpl_orient_mats[0, idx].numpy()
        smpl_x = R_smpl[:, 0]  # X轴方向
        arrow_smpl = create_arrow(origin, smpl_x, [0, 0, 1])
        vis.add_geometry(arrow_smpl)
        
        # 机器人方向 (红色 - X轴)
        R_robot = robot_orient_mats[0, idx].numpy()
        robot_x = R_robot[:, 0]
        arrow_robot = create_arrow(origin, robot_x, [1, 0, 0])
        vis.add_geometry(arrow_robot)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    
    vis.run()
    vis.destroy_window()

@hydra.main(version_base=None, config_path="../cfg", config_name="unitree_g1_fitting")
def main(cfg):
    offsets = compute_offsets(cfg)

if __name__ == "__main__":
    main()