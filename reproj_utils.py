import numpy as np
import matplotlib.pyplot as plt

# =========================
# Projection helpers
# =========================
def project_points_from_P(object_points, P, uv_center=None):
    """
    Project 3D object points to image plane using a 3x4 projection matrix P.
    Parameters
    ----------
    object_points : (N,3) ndarray
        3D points in the same world frame used to build P.
    P : (3,4) ndarray
        Projection matrix so that x ~ P [X,Y,Z,1]^T, in *metric image units* (e.g., mm).
    uv_center : (2,) array-like or None
        If your P was estimated with centered image coordinates (e.g., principal-point subtracted),
        pass the (x0,y0) you want to *add back* for visualization/numeric comparison.
    Returns
    -------
    proj_2d : (N,2) ndarray
        Reprojected image points (same metric as P), with uv_center added if provided.
    """
    obj_h = np.hstack([object_points, np.ones((len(object_points), 1), dtype=object_points.dtype)])
    proj = (P @ obj_h.T)
    proj /= proj[2:3, :]
    proj_2d = proj[:2, :].T
    if uv_center is not None:
        uv_center = np.asarray(uv_center).reshape(1, 2)
        proj_2d = proj_2d + uv_center
    return proj_2d


def project_points_from_KRt(object_points, K, R, t):
    """
    Project 3D object points using intrinsics K and extrinsics (R,t).
    K: 3x3, R: 3x3, t: (3,1) or (3,)
    Returns (N,2) projected points in the same metric as K (e.g., mm on image).
    """
    object_points = np.asarray(object_points, dtype=float)
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError("object_points must be (N,3)")
    K = np.asarray(K, dtype=float)
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3, 1)

    Xw_h = np.hstack([object_points, np.ones((len(object_points), 1))])  # (N,4)
    RT = np.hstack([R, t])  # 3x4
    P = K @ RT              # 3x4
    return project_points_from_P(object_points, P, uv_center=None)


# =========================
# Error computation
# =========================
def reprojection_errors(image_points, reprojected_points):
    """
    Compute per-point reprojection error vectors and summary stats.
    Parameters
    ----------
    image_points : (N,2) ndarray
    reprojected_points : (N,2) ndarray
    Returns
    -------
    err_vec : (N,2) ndarray
        dx, dy (reprojected - original) in image units (e.g., mm)
    err_norm : (N,) ndarray
        Euclidean magnitude of error per point
    stats : dict
        {"rmse": float, "mean": float, "median": float, "max": float}
    """
    image_points = np.asarray(image_points, dtype=float)
    reprojected_points = np.asarray(reprojected_points, dtype=float)
    if image_points.shape != reprojected_points.shape:
        raise ValueError("Shape mismatch between image_points and reprojected_points")
    err_vec = reprojected_points - image_points
    err_norm = np.linalg.norm(err_vec, axis=1)
    stats = {
        "rmse": float(np.sqrt(np.mean(err_norm**2))),
        "mean": float(np.mean(err_norm)),
        "median": float(np.median(err_norm)),
        "max": float(np.max(err_norm)),
    }
    return err_vec, err_norm, stats


# =========================
# Unified plotting
# =========================
def plot_reprojection(image_points, reprojected_points, title="Reprojection",
                      units="mm", show_labels=True, invert_y=False):
    import numpy as np
    import matplotlib.pyplot as plt

    image_points = np.asarray(image_points, dtype=float)
    reprojected_points = np.asarray(reprojected_points, dtype=float)

    fig, ax = plt.subplots()

    # 視窗名稱
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass

    # 繪點
    ax.scatter(image_points[:, 0], image_points[:, 1], s=24)
    ax.scatter(reprojected_points[:, 0], reprojected_points[:, 1], s=24, marker="x")

    # 誤差向量
    for i in range(len(image_points)):
        x0, y0 = image_points[i]
        x1, y1 = reprojected_points[i]
        ax.plot([x0, x1], [y0, y1], linestyle="--", linewidth=0.8)
        if show_labels:
            ax.text(x0, y0, f"{i+1}", fontsize=7)

    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"y ({units})")
    ax.grid(True)

    # ---- 強制座標在 ±200，並指定刻度，避免被等比例或自動刻度改動 ----
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_xticks(np.arange(-200, 201, 50))
    ax.set_yticks(np.arange(-200, 201, 50))

    # 用 aspect='equal' 但不改變既有的座標範圍
    ax.set_aspect('equal', adjustable='box')

    # y 方向是否向下
    if invert_y:
        ax.invert_yaxis()

    # 統計（只印終端機 + 圖下方一行）
    from reproj_utils import reprojection_errors
    _, _, stats = reprojection_errors(image_points, reprojected_points)
    n_points = len(image_points)
    rmse_val = stats['rmse']

    # 終端機輸出
    print(f"=== Reprojection Stats ({title}) ===")
    print(f"RMSE  : {rmse_val:.4f} {units}")
    print(f"Mean  : {stats['mean']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"Max   : {stats['max']:.4f}")
    print("==================================")

    
    # 拉大底邊距，避免擠壓 caption
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.22)

    # 圖下方置中說明
    caption_text = f"Total points: {n_points}    RMSE = {rmse_val:.4f} {units}"
    # 把 caption 放在更高的位置並置中
    fig.text(0.5, 0.05, caption_text, ha="center", va="center", fontsize=10)

    # 給 x 軸標籤多一點下方空間
    ax.set_xlabel(f"x ({units})", labelpad=14)


    

    plt.show()


def plot_fscan(f_values, errors, best_f=None, title="Focal Scan (reprojection error)", units="mm"):
    """
    Line plot for focal-length scan results.
    errors should be *mean* or *RMSE* reprojection errors in 'units'.
    """
    f_values = np.asarray(f_values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    plt.figure()
    plt.plot(f_values, errors, marker="o")
    if best_f is not None:
        plt.axvline(best_f, linestyle="--", label=f"Best f = {best_f:.2f} mm")
        plt.legend()
    plt.xlabel("Focal length (mm)")
    plt.ylabel(f"Reprojection error ({units})")
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    plt.show()
