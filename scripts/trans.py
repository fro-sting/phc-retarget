import numpy as np
from scipy.spatial.transform import Rotation

def clean_matrix_string(matrix_str):
    """辅助函数：清理您输入的带空格的矩阵字符串"""
    lines = matrix_str.strip().replace('[', '').replace(']', '').split('\n')
    cleaned_lines = []
    for line in lines:
        parts = line.strip().split()
        cleaned_lines.append([float(p) for p in parts])
    return np.array(cleaned_lines)

def find_closest_relative_rotation(R1, R2, order='xyz'):
    """
    计算 R1 和 R2 之间最接近的"纯旋转"关系 R_rel，
    使得 R1 * R_rel 约等于 R2。
    即：R1 @ R_rel = R2
    所以：R_rel = R1.T @ R2
    
    此函数会处理 R1 或 R2 是反射矩阵（行列式为-1）的情况。
    """
    
    # 1. 打印输入矩阵的行列式，用于诊断
    det_R1 = np.linalg.det(R1)
    det_R2 = np.linalg.det(R2)
    print(f"--- 诊断信息 ---")
    print(f"R1 的行列式: {det_R1:.4f}")
    print(f"R2 的行列式: {det_R2:.4f}")
    
    if np.allclose(det_R1, 1) and np.allclose(det_R2, 1):
        print("两个输入矩阵都是有效的旋转矩阵。")
    else:
        print("警告：一个或两个输入矩阵不是有效的旋转矩阵（行列式不为1）。")
        print("这可能是一个反射（left-handed）矩阵。\n")

    # 2. 计算相对旋转矩阵 R_rel
    # R1 @ R_rel = R2
    # R_rel = R1.T @ R2
    R_rel_matrix = R1.T @ R2
    
    det_R_rel = np.linalg.det(R_rel_matrix)
    print(f"计算出的 R_rel 的行列式: {det_R_rel:.4f}")
    
    # 3. 修正 R_rel (如果它是一个反射矩阵)
    corrected_R_rel_matrix = R_rel_matrix
    
    if det_R_rel < 0:
        print("R_rel 的行列式为负。这是一个反射变换。")
        print("正在使用 SVD 寻找最接近的""矩阵...\n")
        
        # 使用 SVD (奇异值分解): R_rel = U * S * Vh
        try:
            U, S, Vh = np.linalg.svd(R_rel_matrix)
            
            # 最接近的"纯旋转"矩阵 R' = U @ Vh
            corrected_R_rel_matrix = U @ Vh
            
            # 检查 U @ Vh 的行列式
            if np.linalg.det(corrected_R_rel_matrix) < 0:
                # 如果还是 -1，翻转 U 的最后一列
                U[:, 2] *= -1
                corrected_R_rel_matrix = U @ Vh
                
            print("已修正 R_rel 矩阵，使其行列式为 +1。")
            print(f"修正后的 R_rel 行列式: {np.linalg.det(corrected_R_rel_matrix):.4f}\n")
            
        except np.linalg.LinAlgError:
            print("SVD 计算失败。")
            return None, None

    corrected_R_rel_matrix = corrected_R_rel_matrix.T
    # 4. 将 (可能已修正的) 旋转矩阵转换为欧拉角
    try:
        r = Rotation.from_matrix(corrected_R_rel_matrix)
        euler_angles_deg = r.as_euler(order, degrees=True)
        euler_angles_rad = r.as_euler(order, degrees=False)
        
        # 转换为 π 的倍数形式
        euler_angles_pi = euler_angles_rad / np.pi
        
        return corrected_R_rel_matrix, euler_angles_pi
        
    except ValueError as e:
        print(f"将矩阵转换为欧拉角时出错: {e}")
        return None, None

# --- 主程序 ---

R2 = np.array( [[-0.3459764 ,  0.5804352,  -0.73715353],
                 [-0.38208506, -0.80473727, -0.45432252],
                 [-0.85691965,  0.12447046,  0.5001956 ]])

R1 = np.array(  [[-0.11556382, -0.21839523 ,-0.9689936 ],
 [-0.09393484, -0.9687547 ,  0.22954443],
 [-0.9888485 ,  0.11754931,  0.0914381 ]])

print("====== 求解 R1 @ R_rel = R2 ======\n")
print(f"R1:\n{R1}\n")
print(f"R2:\n{R2}\n")

euler_order = 'xyz' 

# 计算相对旋转
R_rel, euler_angles = find_closest_relative_rotation(R1, R2, order=euler_order)

if euler_angles is not None:
    print(f"\n--- 计算结果 ---")
    print(f"R_rel 矩阵:\n{R_rel}\n")
    
    # 格式化输出，显示为 π 的形式
    euler_str = "["
    for i, angle in enumerate(euler_angles):
        if np.isclose(angle, 0):
            euler_str += "0"
        else:
            euler_str += f"{angle:.4f}π"
        if i < len(euler_angles) - 1:
            euler_str += ", "
    euler_str += "]"
    print(f"对应的欧拉角 (顺序: '{euler_order}'):\n{euler_str}\n")

    # 验证结果
    R2_reconstructed = R1 @ R_rel
    
    print("--- 验证 ---")
    print(f"R1 @ R_rel (重构):\n{R2_reconstructed}\n")
    print(f"目标 R2:\n{R2}\n")
    
    # 计算重构误差
    error = np.linalg.norm(R2_reconstructed - R2)
    print(f"重构误差 (Frobenius范数): {error:.6f}")
    
    # 计算两个矩阵之间的旋转角度误差
    R_error = R2_reconstructed.T @ R2
    trace = np.trace(R_error)
    angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    print(f"旋转角度误差: {np.degrees(angle_error):.6f}°")