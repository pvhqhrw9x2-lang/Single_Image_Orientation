import numpy as np
import cv2
import matplotlib.pyplot as plt
from R_decom import rotation_matrix_to_angles

# 新增：統一的重投影/繪圖工具
from reproj_utils import (
    project_points_from_KRt,
    reprojection_errors,
    plot_reprojection,
    plot_fscan
)

def estimate_focal_length(obj_pts_raw, img_pts_raw, uv_center=None):
    """
    obj_pts_raw : (N,3) 世界座標（mm）
    img_pts_raw : (N,2) 像點（mm），原點在影像中心或左上皆可；若不是中心，就把 uv_center 傳進來
    uv_center   : (2,) 若 img_pts_raw 不是中心化，請傳 (x0,y0)（mm）；若已中心化則傳 None
    """

    print("\n=== Detect Possible Focal Length ===")
    print(f"Loaded {len(obj_pts_raw)} 3D points, {len(img_pts_raw)} 2D points...\n")

    # ---- 0) 中心化像點（和 DLT 一致）----
    if uv_center is None:
        img_c = img_pts_raw.astype(np.float32)  # 已中心化
        uv_center = np.zeros(2, dtype=np.float32)
    else:
        uv_center = np.asarray(uv_center, np.float32).reshape(2)
        img_c = (img_pts_raw - uv_center).astype(np.float32)  # 中心化像點（mm）

    # ---- 1) 3D 平移/尺度歸一（你原本就有做；與投影無衝突）----
    obj_origin = obj_pts_raw[0]
    obj_pts_norm = (obj_pts_raw - obj_origin) / 1000.0  # 縮放不影響 PnP 幾何
    obj = obj_pts_norm.astype(np.float32).reshape(-1,1,3)
    img = img_c.astype(np.float32).reshape(-1,1,2)

    distCoeffs = np.zeros(4, dtype=np.float32)

    # ---- 2) 統一：K 主點設 0（因為我們已中心化）----
    def reprojection_error_mean(focal_length):
        K = np.array([[focal_length, 0, 0],
                      [0, focal_length, 0],
                      [0, 0, 1]], dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return np.inf
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, distCoeffs)  # 這裡投影結果是「中心化座標」
        proj = proj.reshape(-1,2)
        # 注意：和中心化的 img_c 比
        err = np.linalg.norm(proj - img_c, axis=1)
        return float(np.mean(err))

    def scan_focal_range(f_start, f_end, step):
        f_candidates = np.arange(f_start, f_end + 1e-9, step, dtype=float)
        errors = []
        for f in f_candidates:
            errors.append(reprojection_error_mean(f))
        return f_candidates, np.array(errors, dtype=float)

    # ---- 3) 掃 f ----
    try:
        f_start = float(input("Please enter f range min (mm) : "))
        f_end   = float(input("Please enter f range max (mm) : "))
        f_step  = float(input("Please enter step (mm) : "))
    except Exception:
        print("Input Error, offset range:1~100 mm, step = 1 mm")
        f_start, f_end, f_step = 1, 100, 1

    print(f"\n=== coarse scan : {f_start} ~ {f_end} mm, step = {f_step} ===")
    f_coarse, e_coarse = scan_focal_range(f_start, f_end, f_step)
    i_best_c = int(np.argmin(e_coarse))
    best_f_c = float(f_coarse[i_best_c])
    print(f"\nBest focal (coarse) = {best_f_c:.1f} mm (Mean error = {e_coarse[i_best_c]:.6f})")

    fine_start = max(f_start, best_f_c - 10)
    fine_end   = min(f_end,   best_f_c + 10)

    print(f"\n=== fine scan : {fine_start} ~ {fine_end} mm, step = 0.1 ===")
    f_fine, e_fine = scan_focal_range(fine_start, fine_end, 0.1)
    i_best_f = int(np.argmin(e_fine))
    best_f   = float(f_fine[i_best_f])
    print(f"\nBest focal (fine) = {best_f:.2f} mm (Mean error = {e_fine[i_best_f]:.6f})")

    # ---- 4) 用 best f 解最終 PnP，並「在中心化座標系」做重投影 ----
    K_best = np.array([[best_f, 0, 0],
                       [0, best_f, 0],
                       [0, 0, 1]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K_best, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed at best focal.")

    R_wc, _ = cv2.Rodrigues(rvec)     # OpenCV：world→camera
    t_wc    = tvec.reshape(3,1)

    # camera center in the *normalized world frame* (meters)
    C_norm_m = (-R_wc.T @ t_wc).flatten()  # world frame（歸一化尺度）
    # 換回原始世界座標（mm）
    camera_center_original_mm = C_norm_m * 1000.0 + obj_origin

    # ---- 5) 與 DLT 完全一致的「中心化→投影→加回中心」畫圖 ----
    #   先在中心化座標做投影，再加回 uv_center 與原始像點比較
    proj_c = project_points_from_KRt(object_points=obj_pts_norm, K=K_best, R=R_wc, t=t_wc)  # (N,2) 中心化
    proj   = proj_c + uv_center  # 加回主點（與 img_pts_raw 同系）

    # 統一畫法 + 誤差統計
    plot_reprojection(
        image_points=img_pts_raw,
        reprojected_points=proj,
        title="PnP Reprojection (best f)",
        units="mm",
        show_labels=True,
        invert_y=False  # 若你的座標是 y 向下為正，把這裡改 True（DLT 也一樣）
    )
    _, _, stats = reprojection_errors(img_pts_raw, proj)
    print("Reprojection Error Stats (best f):", stats)

    # ---- 6) 用統一曲線圖畫掃描結果 ----
    plot_fscan(f_coarse, e_coarse, best_f=best_f_c, title="Coarse Focal Scan", units="mm")
    plot_fscan(f_fine,   e_fine,   best_f=best_f,   title="Fine Focal Scan (±10mm)", units="mm")

    # ---- 7) 角度（以你原先的 rotation_matrix_to_angles）----
    #   注意：你要顯示 camera→world 的歐拉角時，請用 R_cw = R_wc.T
    R_cw = R_wc.T
    euler_angles = rotation_matrix_to_angles(R_cw)

    print("\n=== PnP Estimated EOP ===")
    print(f"Omega, Phi, Kappa (deg) : [{euler_angles[0]:.6f}, {euler_angles[1]:.6f}, {euler_angles[2]:.6f}]")
    print(f"Camera Position X (m) : [{camera_center_original_mm[0]/1000:.6f}, "
          f"{camera_center_original_mm[1]/1000:.6f}, {camera_center_original_mm[2]/1000:.6f}]")

    return best_f, float(e_fine[i_best_f])
