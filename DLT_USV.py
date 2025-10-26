import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from R_decom import rotation_matrix_to_angles
from R_decom import as_euler
from reproj_utils import project_points_from_P, plot_reprojection


def DLT_USV(xyz, uv):

    """
    Direct Linear Transformation based on Förstner method (homogeneous system).
    Inputs:
        xyz - (n, 3) array of 3D points
        uv  - (n, 2) array of 2D image coordinates
    Returns:
        P - (3, 4) projection matrix
    """
    
    uv_center = np.mean(uv, axis=0)
    uv_centered = uv - uv_center
    # print("Mean image point (before projection):", uv_center)

    # check if imported points are matched
    m = uv.shape[0]
    n = xyz.shape[0]
    if m != n:
        raise ValueError("Number of 2D and 3D points must match.")
    if m < 6 | n < 6:
        raise ValueError("At least 6 points are required.")
    
    # centerized 3D points
    scale = 1
    xc = np.mean(xyz, axis=0, dtype=np.float64).reshape(3,1)
    xyz_centered = (xyz - xc.T) / scale
    
    # design matrix
    A = []
    for i in range(n):
        X, Y, Z = xyz_centered[i, 0], xyz_centered[i, 1], xyz_centered[i, 2]
        u, v = uv_centered[i, 0], uv_centered[i, 1]
        row1 = [-X, -Y, -Z, -1, 0, 0, 0, 0, u*X, u*Y, u*Z, u]
        row2 = [0, 0, 0, 0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v]
        A.append(row1)
        A.append(row2)
    A = np.array(A)
    A_L = A[:, :11]

    # SVD solution: Ax = 0 
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(12, 1)
    P = P/P[-1] 
    P = P.reshape(3,4).astype(np.float64)

    L = la.cholesky(A_L.T @ A_L, lower=True)
    uv_L = uv_centered.flatten().reshape(2*n, 1).astype(np.float64)
    y_L = la.solve_triangular(L, A_L.T @ uv_L, lower=True, overwrite_b=False)
    P_L = la.solve_triangular(L.T, y_L, lower=False, overwrite_b=False)

    # print(P)
    # norms = np.linalg.norm(P, axis=1)
    # print("Row norms:", norms)     

    # decomposite P matrix

    cam_est = P_Foerstner(P, xc, uv_center, scale)
    # cam_est = P_Luhmann(P_L, xc, uv_center, scale)

    proj_2d = project_points_from_P(xyz_centered, cam_est['P'], uv_center=uv_center)
    plot_reprojection(
        image_points=uv,                 # 原始像點 (mm)
        reprojected_points=proj_2d,      # 重投影點 (mm)
        title="DLT Reprojection",
        units="mm",
        show_labels=True,
        invert_y=False                   # 若你的座標系 y 向下為正，這裡改 True
    )        
    return cam_est

def rq_decomposition(M):
    from scipy.linalg import qr
    M = np.asarray(M)
    M_flip = np.flipud(np.fliplr(M))
    Q, R = qr(M_flip.T)
    R = np.fliplr(np.flipud(R.T))
    Q = np.fliplr(np.flipud(Q.T))
    return R, Q

def P_Foerstner(P, xc, uv_center, scale):

    M = P[:, :3]
    p4 = P[:, 3].reshape(3, 1)

    # RQ decomposition of M
    K, R = rq_decomposition(M)
    
    # Normalize K to make diagonal positive
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R
    R = R.T
    if R[2, 2] < 0:
        R = R @ np.diag([1, 1, -1]) 

    # # U, _, Vt = np.linalg.svd(R)
    # R_ortho = U @ Vt  # 正交矩陣
    # R = R_ortho.T     # 轉置符合後續使用

    # Normalize K (optional, e.g., K[2,2] = 1)
    K = K / K[2, 2]

    # Camera center
    C = -np.linalg.inv(M) @ p4
    X = C.flatten()* scale + xc.flatten()

    x0_est = K[0, 2]+ uv_center[0]
    y0_est = K[1, 2]+ uv_center[1]
    cx = K[0, 0]
    cy = K[1, 1]
    c = np.mean([cx, cy])
#     L = -1/np.sqrt(P[8,0]**2+P[9,0]**2+P[10,0]**2)
#     x0_est = L**2*(P[0,0]*P[8,0]+P[1,0]*P[9,0]+P[2,0]*P[10,0])
#     y0_est = L**2*(P[4,0]*P[8,0]+P[5,0]*P[9,0]+P[6,0]*P[10,0])
#     cx = np.sqrt(L**2*(P[0,0]**2+P[1,0]**2+P[2,0]**2)-x0_est**2)
#     cy = np.sqrt(L**2*(P[4,0]**2+P[5,0]**2+P[6,0]**2)-y0_est**2)

#     # R matrix
#     r11 = L*(x0_est*P[8,0]-P[0,0])/cx
#     r12 = L*(y0_est*P[8,0]-P[4,0])/cy
#     r13 = L*P[8,0]
#     r21 = L*(x0_est*P[9,0]-P[1,0])/cx
#     r22 = L*(y0_est*P[9,0]-P[5,0])/cy
#     r23 = L*P[9,0]
#     r31 = L*(x0_est*P[10,0]-P[2,0])/cx
#     r32 = L*(y0_est*P[10,0]-P[6,0])/cy
#     r33 = L*P[10,0]
#     R = np.array([[r11, r12, r13],
#                   [r21, r22, r23],
#                   [r31, r32, r33]])
    # R = R.T
    # check if R.T * R = I
    # print('R')
    # print(R)
    # U = R.T @ R
    # print('U')
    # print(U)


    pos = rotation_matrix_to_angles(R)
        
#     A = np.array([[P[0,0], P[1,0], P[2,0]], 
#               [P[4,0], P[5,0], P[6,0]], 
#               [P[8,0], P[9,0], P[10,0]]], dtype=np.float64)
#     b = -np.array([P[3,0], P[7,0], 1], dtype=np.float64).reshape(3,1)

#    # SVD decomposition: A = U Σ V^T
#     U, S, VT = np.linalg.svd(A, full_matrices=False)

#     # Invert Σ (singular values)
#     S_inv = np.diag(1 / S)

#     # Compute pseudo-inverse A⁺ = V Σ⁻¹ Uᵀ
#     A_pseudo_inv = VT.T @ S_inv @ U.T

#     # Solve for X_vec
#     X_vec = A_pseudo_inv @ b  # shape (3, 1)
#     X = np.array([X_vec[0,0] + xc[0,0], X_vec[1,0] + xc[1,0], X_vec[2,0] + xc[2,0]], dtype=np.float64)

    # X = np.asarray(-la.inv([[P[0], P[1], P[2]], [P[4], P[5], P[6]], [P[8], P[9], P[10]]]) @ np.array([P[3], P[7], 1]).reshape(-1, 1))
    # # X = la.solve(np.array([[P[0], P[1], P[2]], [P[4], P[5], P[6]], [P[8], P[9], P[10]]]), -np.array([P[3], P[7], 1]))
    # X = np.array([X[0]+xc[0], X[1]+xc[1], X[2]+xc[2]])

    print('\n=== DLT Result ===')
    print(f"Focal Length f: {c:.6f} mm")
    print(f"Image Center x0: [{x0_est:.6f}, {y0_est:.6f}] mm")
    # print("Rotation Matrix R:\n", R)
    print(f"Omega, Phi, Kappa (deg) : [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
    print(f"Camera Position X (m) : [{X[0]/1000:.6f}, {X[1]/1000:.6f}, {X[2]/1000:.6f}]")

    # save_results("dlt_Bild3.txt", P, U, cx, cy, x0_est, y0_est, X, R, pos)
    # compare_files("dlt_Bild3.txt", "example__Bild3.txt", "comparison_Bild3.txt")

    return {
        'P': P,
        'c': c,
        'x0_est': x0_est,
        'y0_est': y0_est,
        'R': R,
        'position': X,
        'rotation_angles': pos
    }

def P_Luhmann(P, xc, uv_center, scale):
    
    P = P.flatten().astype(np.float64)
    L = -1/np.sqrt(P[8]**2+P[9]**2+P[10]**2)
    x0_est = L**2*(P[0]*P[8]+P[1]*P[9]+P[2]*P[10])
    y0_est = L**2*(P[4]*P[8]+P[5]*P[9]+P[6]*P[10])
    cx = np.sqrt(L**2*(P[0]**2+P[1]**2+P[2]**2)-x0_est**2)
    cy = np.sqrt(L**2*(P[4]**2+P[5]**2+P[6]**2)-y0_est**2)
    c = np.mean([cx, cy])

    # R matrix
    r11 = L*(x0_est*P[8]-P[0])/cx
    r12 = L*(y0_est*P[8]-P[4])/cy
    r13 = L*P[8]
    r21 = L*(x0_est*P[9]-P[1])/cx
    r22 = L*(y0_est*P[9]-P[5])/cy
    r23 = L*P[9]
    r31 = L*(x0_est*P[10]-P[2])/cx
    r32 = L*(y0_est*P[10]-P[6])/cy
    r33 = L*P[10]
    # L = -1 / np.linalg.norm(P[2, :3])
    # x0_est = L**2 * np.dot(P[0, :3], P[2, :3])
    # y0_est = L**2 * np.dot(P[1, :3], P[2, :3])
    # cx = np.sqrt(L**2 * np.dot(P[0, :3], P[0, :3]) - x0_est**2)
    # cy = np.sqrt(L**2 * np.dot(P[1, :3], P[1, :3]) - y0_est**2)
    # c = np.mean([cx, cy])

    # # R matrix
    # r11 = L * (x0_est * P[2, 0] - P[0, 0]) / cx
    # r12 = L * (y0_est * P[2, 0] - P[1, 0]) / cy
    # r13 = L * P[2, 0]
    # r21 = L * (x0_est * P[2, 1] - P[0, 1]) / cx
    # r22 = L * (y0_est * P[2, 1] - P[1, 1]) / cy
    # r23 = L * P[2, 1]
    # r31 = L * (x0_est * P[2, 2] - P[0, 2]) / cx
    # r32 = L * (y0_est * P[2, 2] - P[1, 2]) / cy
    # r33 = L * P[2, 2]
    R = np.array([[r11, r12, r13],
                  [r21, r22, r23],
                  [r31, r32, r33]])
    detR = np.linalg.det(R)
    if detR < 0:
        R = -R
        P = -P
        # L = -L
    L = -1/np.sqrt(P[8]**2+P[9]**2+P[10]**2)
    x0_est = L**2*(P[0]*P[8]+P[1]*P[9]+P[2]*P[10])
    y0_est = L**2*(P[4]*P[8]+P[5]*P[9]+P[6]*P[10])
    cx = np.sqrt(L**2*(P[0]**2+P[1]**2+P[2]**2)-x0_est**2)
    cy = np.sqrt(L**2*(P[4]**2+P[5]**2+P[6]**2)-y0_est**2)
    c = np.mean([cx, cy])
    # R = R.T
    # check if R.T * R = I
    U = R.T @ R
    print('U')
    print(U)

    pos = rotation_matrix_to_angles(R)

    X = np.asarray(-la.inv([[P[0], P[1], P[2]], [P[4], P[5], P[6]], [P[8], P[9], P[10]]]) @ np.array([P[3], P[7], 1]).reshape(-1, 1))
    # X = la.solve(np.array([[P[0], P[1], P[2]], [P[4], P[5], P[6]], [P[8], P[9], P[10]]]), -np.array([P[3], P[7], 1]))
    X = X.flatten()*scale + xc.flatten()
    x0_est = x0_est + uv_center[0]
    y0_est = y0_est + uv_center[1]    

    print('\n=== DLT Result ===')
    print("Focal Length f: ", c, " mm")
    print("Image Center x0: ", x0_est, y0_est, " mm")
    print("Rotation Matrix R:\n", R)
    print(f"Omega, Phi, Kappa (deg) : [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
    print(f"Camera Position X (m) : [{X[0]/scale:.9f}, {X[1]/scale:.9f}, {X[2]/scale:.9f}]")
    return {
        'P': P,
        'c': c,
        'x0_est': x0_est,
        'y0_est': y0_est,
        'R': R,
        'position': X,
        'rotation_angles': pos
    }

def visualize_reprojection(object_points, image_points, uv_center, P, title="DLT Reprojection"):
    
    obj_h = np.hstack([object_points, np.ones((len(object_points), 1))])
    proj = P @ obj_h.T
    proj /= proj[2, :]
    proj_2d = proj[:2, :].T+uv_center

    plt.figure()
    plt.scatter(image_points[:, 0], image_points[:, 1], c='blue', label='Image Points')
    plt.scatter(proj_2d[:, 0], proj_2d[:, 1], c='red', marker='x', label='Reprojected Points')

    for i in range(len(image_points)):
        plt.plot([image_points[i, 0], proj_2d[i, 0]],
                 [image_points[i, 1], proj_2d[i, 1]], 'k--', linewidth=0.5)

    # plt.gca().invert_yaxis()
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_points_with_principal_point(image_points, x0, y0):
    plt.figure(figsize=(6, 6))
    plt.scatter(image_points[:, 0], image_points[:, 1], c='blue', label='Image Points')
    plt.scatter(x0, y0, c='red', marker='x', s=100, label='Principal Point')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    # plt.gca().invert_yaxis()
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.title("Image Points and Estimated Principal Point")
    plt.show()

# def compare_results(result1, result2):
#     comparison = {}
    
#     comparison['P_diff'] = result2['P'] - result1['P']
#     comparison['cx_diff'] = result2['cx'] - result1['cx']
#     comparison['cy_diff'] = result2['cy'] - result1['cy']
#     comparison['x0_diff'] = result2['x0_est'] - result1['x0_est']
#     comparison['y0_diff'] = result2['y0_est'] - result1['y0_est']
#     comparison['position_diff'] = result2['position'] - result1['position']
#     comparison['rotation_diff'] = result2['rotation_angles'] - result1['rotation_angles']
    
#     return comparison



# def save_results(filename, dlt_matrix, U, cx, cy, x0_est, y0_est, X, R, POS):
#     with open(filename, "w") as f:
#         f.write("Final result:\n")
#         f.write("="*73 + "\n")
#         f.write("DLT Parameters\n")
#         for i, val in enumerate(dlt_matrix, 1):
#             f.write(f" L{i} = {val:.10f}\n")
        
#         f.write("\nOrientation parameters:\n")
#         f.write(f"X0   = {X[0]:>15.6f}\n")
#         f.write(f"Y0   = {X[1]:>15.6f}\n")
#         f.write(f"Z0   = {X[2]:>15.6f}\n")
#         f.write(f"omega= {-np.atan2(R[1, 2], R[2, 2]):.6f} ({POS[0]:.5f})\n")
#         f.write(f"  phi= {np.arcsin(R[0, 2]):.6f} ({POS[1]:.5f})\n")
#         f.write(f"kappa= {-np.atan2(R[0, 1], R[0, 0]):.6f} ({POS[2]:.5f})\n")
        
#         f.write("\nR =\n")
#         for row in R:
#             f.write(" ".join(f"{val: .8f}" for val in row) + "\n")
        
#         f.write("\nR(t)R =\n")
#         for row in U:
#             f.write(" ".join(f"{val: .8f}" for val in row) + "\n")
        
#         f.write("\nInterior orientation\n")
#         f.write(f"c   = { -(cx+cy)/2:.6f}\n")
#         f.write(f"x'o = {x0_est:.6f}\n")
#         f.write(f"y'o = {y0_est:.6f}\n")

# def extract_numbers(line):
#     return [float(num) for num in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)]

# def compare_files(file1, file2, output_file):
#     with open(file1, "r") as f1, open(file2, "r") as f2:
#         lines1 = f1.readlines()
#         lines2 = f2.readlines()
    
#     differences = []
#     for i, (line1, line2) in enumerate(zip(lines1, lines2)):
#         nums1 = extract_numbers(line1)
#         nums2 = extract_numbers(line2)
        
#         if nums1 and nums2 and len(nums1) == len(nums2):
#             diffs = [abs(a - b) for a, b in zip(nums1, nums2)]
#             differences.append(f"Referenced: {line2.strip()}\nResult IFP:      {line1.strip()}\n{diffs}\n")
    
#     with open(output_file, "w") as f:
#         if differences:
#             f.write("Differences found:\n\n" + "\n".join(differences))
#         else:
#             f.write("No significant differences found. Files are nearly identical.")


# P1, xc1 = DLT_USV(3, xyz, uv)
    # result1 = process_dlt_results(P1, xc1)
    
    # P2, xc2 = DLT(3, xyz, uv2)
    # result2 = process_dlt_results(P2, xc2)
    
    # reprojection error
    # error1 = compute_reprojection_error(xyz, uv1, P1, xc1)
    # error2 = compute_reprojection_error(xyz, uv2, P2, xc2)
    
    # comparison = compare_results(result1, result2)
    
    # print("\nSelected Points:")
    # print("DLT:\n", result1['P'])
    # print("X:", result1['position'].T)
    # # print("diff from original:", result1['position'].T-np.array([693554897.049, 5677009899.994, 124725.393]))
    # print("POS(omega, phi, kappa):", result1['rotation_angles'])
    # print("c:", result1['c'])
    # print("(x0, y0):", result1['x0_est'], result1['y0_est'])
    # # print("avg projection error:", np.mean(error1))
    
    # print("\nShifted Points:")
    # print("DLT:\n", result2['P'])
    # print("X:", result2['position'].T)
    # print("diff from original:", result2['position'].T-np.array([693554897.049, 5677009899.994, 124725.393]))
    # print("POS(omega, phi, kappa):", result2['rotation_angles'])
    # print("f (cx, cy):", result2['cx'], result2['cy'])
    # print("(x0, y0):", result2['x0_est'], result2['y0_est'])
    # # print("avg projection error:", np.mean(error2))
    
    # print("\nComparison:")
    # print("DLT:\n", comparison['P_diff'])
    # print("X:", comparison['position_diff'].T)
    # print("POS:", comparison['rotation_diff'])
    # print("f (cx, cy):", comparison['cx_diff'], comparison['cy_diff'])
    # print("(x0, y0):", comparison['x0_diff'], comparison['y0_diff'])
    
    # pos_change_percent = np.abs(comparison['position_diff'] / result1['position']) * 100
    # print("\n:", pos_change_percent, "%")
    
    # rot_change_degrees = np.abs(comparison['rotation_diff'])
    # print(":", rot_change_degrees)
    # print('DLT Matrix:')
    # print(P)
    # print('\nReprojection Error:')
    # print(err)

    # plot_points_with_principal_point(uv, 5.56, -98.01)
    # print("\nDLT Estimated:")
    # print("P matrix:", cam_est['P'])
    # print("X:", cam_est['position'].T)
    # # print("diff from original:", result1['position'].T-np.array([693554897.049, 5677009899.994, 124725.393]))
    # print("POS(omega, phi, kappa):", cam_est['rotation_angles'])
    # print("c:", cam_est['c'])
    # print("(x0, y0):", cam_est['x0_est'], cam_est['y0_est'])
    # # print("avg projection error:", np.mean(error1))
