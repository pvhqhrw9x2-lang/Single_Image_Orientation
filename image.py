import cv2
import numpy as np
import csv
import os
import string
import argparse
import time

# ================= User Config =================
IMG_PATH = "BLDAM_MBA-4-b-24_141_8.tif"       # 可用 --image 覆蓋
OUTPUT_CSV = "image_points_8.csv"   # 可用 --out 覆蓋
WIDTH_MM = 400.0                  # 實體寬 (mm)
HEIGHT_MM = 400.0                 # 實體高 (mm)
INIT_ZOOM = 4                     # 2~12

# 兩種方式指定初始半徑：像素或毫米 (二擇一)。若用毫米，將 INIT_RADIUS 設 None。
INIT_RADIUS = None                # 例如 60；若用毫米就設 None
INIT_RADIUS_MM = 6.0              # 例如 5 mm 的半徑
RADIUS_MIN_PX = 6
RADIUS_MAX_PX = 1200

SHOW_ZOOM = True                  # 按 m 可切換放大鏡

# Zoom 固定畫布大小 + 十字線粗細（讓 UI 不跟著縮放）
ZOOM_CANVAS = 420                 # 固定正方形畫布大小
CROSS_THICK  = 2                  # 十字線粗細（像素）
ZOOM_LABEL_MAX_FRAC = 0.90        # 文字框最大寬度為畫布的比例

# —— HUD 固定設定（主視窗操作提示，固定左上角與大小）——
HUD_TEXT_PX = 40                  # 固定字高（像素，建議 32/40/48/60）
STATUS_BG_ALPHA = 0.90            # 背景透明度（越大越實）
STATUS_BORDER_THICK = 3           # 邊框粗細
STATUS_AUTOCONTRAST = True        # 自動挑黑/白底與字色，提高對比

# UI 字體
STATUS_FONT_SCALE = 0.9           # 實際會在 main() 依 HUD_TEXT_PX 覆蓋
STATUS_FONT_THICK = 2
ZOOM_FONT_SCALE   = 0.9
ZOOM_FONT_THICK   = 2

# 防雙擊與軸線最小長度
MIN_AXIS_LEN_PX = 20
DOUBLECLICK_TIME_MS = 300
DOUBLECLICK_DIST_PX = 5

# —— Overlay visibility knobs ——
AXIS_THICK = 7                   # 主軸線粗細
AXIS_HALO_THICK = AXIS_THICK + 4
POINT_RADIUS = 30                 # 量測點半徑
POINT_HALO_RADIUS = POINT_RADIUS + 4
LABEL_BG_ALPHA = 0.65            # 標籤底色透明度(0~1)
LABEL_PAD = 4                    # 標籤內距
# —— Point label 外觀（固定像素大小）——
LABEL_TEXT_PX = 40          # 點名目標字高（像素，建議 32/40/48）
LABEL_FONT_THICK = 2        # 字粗
LABEL_OFFSET_X = 14         # 和點的水平位移（像素）
LABEL_OFFSET_Y = -14        # 和點的垂直位移（像素）

# ==============================================

# Windows
WIN_MAIN = "Image"
WIN_ZOOM = "Zoom"

# Colors (BGR)
COLOR_INFO  = (255, 255, 255)
COLOR_BG    = (0, 0, 0)
COLOR_XAXIS = (0, 200, 255)   # LR line
COLOR_YAXIS = (255, 170, 0)   # UD line
COLOR_POINT = (0, 165, 255)
COLOR_CROSS = (0, 255, 255)

# --------- State ---------
axis_points = []          # [Left, Right, Up, Down]
measured_points = []      # list of {'label','xy','rel','mm'}
labels = list(string.ascii_uppercase)
label_index = 0

origin = None             # LR-UD intersection (px)
x_axis_vec = None         # unit vector along LR (px), + toward Right
y_axis_vec = None         # unit vector along UD (px), + toward Up
pixel_to_mm_matrix = None # 2x3 affine px -> mm

zoom_factor = INIT_ZOOM
zoom_radius = INIT_RADIUS if INIT_RADIUS is not None else 60  # 先有個保底
desired_radius_mm = None  # 若使用 mm 指定半徑，記錄目標值
approx_px_per_mm = 1.0    # 載圖後粗估；校正後會更精準
mm_per_px_iso = None      # 完成軸線校正後的平均 mm/px（顯示用）

img = None
img_copy = None
mouse_xy = (0, 0)

# double-click tracker
_last_click_tick = 0
_last_click_pos = None

# toast 提示（短暫訊息）
toast_msg = ""
toast_deadline = 0.0

# ------------- CLI -------------
def parse_args():
    p = argparse.ArgumentParser(description="Point picker + magnifier + 400x400 mm calibration (center 0,0)")
    p.add_argument("--image", "-i", type=str, default=None, help="Path to input image (overrides IMG_PATH)")
    p.add_argument("--out", "-o", type=str, default=None, help="Base name for CSV/PNG (overrides OUTPUT_CSV)")
    p.add_argument("--zoom", type=int, default=None, help="Initial zoom factor (2-12)")
    p.add_argument("--radius", type=int, default=None, help="Initial radius in pixels")
    p.add_argument("--radius-mm", type=float, default=None, help="Initial radius in millimeters")
    p.add_argument("--no-zoom", action="store_true", help="Start with magnifier hidden")
    return p.parse_args()

# ------------- Utilities -------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def unit(v):
    n = np.linalg.norm(v.astype(np.float32))
    return v / n if n > 1e-9 else v

def line_intersection(p1, p2, p3, p4):
    A1 = p2[1] - p1[1]; B1 = p1[0] - p2[0]; C1 = A1 * p1[0] + B1 * p1[1]
    A2 = p4[1] - p3[1]; B2 = p3[0] - p4[0]; C2 = A2 * p3[0] + B2 * p3[1]
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-9: return None
    x = (B2 * C1 - B1 * C2) / det; y = (A1 * C2 - A2 * C1) / det
    return (int(round(x)), int(round(y)))

def build_pixel_to_mm(origin, x_point, y_point, x_mm, y_mm):
    src = np.float32([origin, x_point, y_point])
    dst = np.float32([[0, 0], [x_mm, 0], [0, y_mm]])
    return cv2.getAffineTransform(src, dst)

def pixel_to_mm(pt, M):
    if M is None: return (None, None)
    x, y = pt
    X = M[0,0]*x + M[0,1]*y + M[0,2]
    Y = M[1,0]*x + M[1,1]*y + M[1,2]
    return (float(X), float(Y))

def fmt(v, nd=3):
    return "None" if v is None else f"{v:.{nd}f}"

def toast(msg, dur=1.2):
    global toast_msg, toast_deadline
    toast_msg = msg
    toast_deadline = time.time() + float(dur)

def _is_double_click(x, y):
    global _last_click_tick, _last_click_pos
    t = cv2.getTickCount()
    is_db = False
    if _last_click_pos is not None and _last_click_tick:
        dt_ms = (t - _last_click_tick) / cv2.getTickFrequency() * 1000.0
        dx = x - _last_click_pos[0]; dy = y - _last_click_pos[1]
        if dt_ms < DOUBLECLICK_TIME_MS and (dx*dx + dy*dy) ** 0.5 < DOUBLECLICK_DIST_PX:
            is_db = True
    _last_click_tick = t
    _last_click_pos = (x, y)
    return is_db

def draw_filled_box(img_, x1, y1, x2, y2, bgr=(0,0,0), alpha=0.6):
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(img_.shape[1]-1, x2), min(img_.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1: return
    ov = img_.copy()
    cv2.rectangle(ov, (x1,y1), (x2,y2), bgr, -1)
    cv2.addWeighted(ov, alpha, img_, 1-alpha, 0, img_)

def font_scale_for_height(target_px, thickness=2, sample_text="Hg"):
    (tw, th), bl = cv2.getTextSize(sample_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, thickness)
    h1 = th + bl if (th + bl) > 0 else max(1, th)
    return max(0.1, float(target_px) / h1)

# ------------- Drawing -------------
def draw_text(img_, text, org, scale=STATUS_FONT_SCALE, color=COLOR_INFO, thick=STATUS_FONT_THICK):
    cv2.putText(img_, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img_, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_status_box(lines, alpha=STATUS_BG_ALPHA):
    """固定左上角 HUD（半透明底＋邊框），字級由 STATUS_FONT_SCALE 控制"""
    if not lines: return

    # 量測面板大小
    inner_pad = 10
    widths, heights = [], []
    for line in lines:
        (tw, th), bl = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, STATUS_FONT_SCALE, STATUS_FONT_THICK)
        widths.append(tw); heights.append(th + bl)
    box_w = max(widths) + inner_pad*2
    box_h = sum(heights) + inner_pad*2 + (len(lines)-1)*6

    # 固定左上角（含外距）
    x1, y1 = 20, 20
    x2, y2 = x1 + box_w, y1 + box_h

    # 自動對比：決定黑底/白底與字色
    bg_color = (0,0,0); text_color = (255,255,255)
    if STATUS_AUTOCONTRAST:
        H, W = img_copy.shape[:2]
        sx1, sy1 = max(0, x1), max(0, y1)
        sx2, sy2 = min(W-1, x2), min(H-1, y2)
        roi = img_copy[sy1:sy2, sx1:sx2]
        if roi.size > 0:
            mean = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())
            if mean < 80:
                bg_color, text_color = (255,255,255), (0,0,0)
            else:
                bg_color, text_color = (0,0,0), (255,255,255)

    # 半透明底 + 邊框
    ov = img_copy.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(ov, alpha, img_copy, 1 - alpha, 0, img_copy)
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255,255,255), STATUS_BORDER_THICK, cv2.LINE_AA)

    # 寫文字（黑描邊 + 文字色）
    y = y1 + inner_pad
    for line in lines:
        (tw, th), bl = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, STATUS_FONT_SCALE, STATUS_FONT_THICK)
        draw_text(img_copy, line, (x1 + inner_pad, y + th), scale=STATUS_FONT_SCALE, color=text_color, thick=STATUS_FONT_THICK)
        y += th + bl + 6

def draw_axes_and_origin():
    # LR
    if len(axis_points) >= 2:
        L, R = axis_points[0], axis_points[1]
        cv2.line(img_copy, L, R, (255,255,255), AXIS_HALO_THICK+2, cv2.LINE_AA)
        cv2.line(img_copy, L, R, (0,0,0),       AXIS_HALO_THICK,   cv2.LINE_AA)
        cv2.line(img_copy, L, R, COLOR_XAXIS,   AXIS_THICK,        cv2.LINE_AA)
    # UD
    if len(axis_points) >= 4:
        U, D = axis_points[2], axis_points[3]
        cv2.line(img_copy, U, D, (255,255,255), AXIS_HALO_THICK+2, cv2.LINE_AA)
        cv2.line(img_copy, U, D, (0,0,0),       AXIS_HALO_THICK,   cv2.LINE_AA)
        cv2.line(img_copy, U, D, COLOR_YAXIS,   AXIS_THICK,        cv2.LINE_AA)
    # Origin（三層）
    if origin is not None:
        cv2.drawMarker(img_copy, origin, (255,255,255), cv2.MARKER_TILTED_CROSS, 28, 4)
        cv2.drawMarker(img_copy, origin, (0,0,0),       cv2.MARKER_TILTED_CROSS, 24, 3)
        cv2.drawMarker(img_copy, origin, (0,255,0),     cv2.MARKER_TILTED_CROSS, 20, 2)

def draw_points_labels():
    h, w = img_copy.shape[:2]
    for p in measured_points:
        x, y = p['xy']
        lab = p['label']

        # 點：白 halo + 黑 halo + 彩色本體
        cv2.circle(img_copy, (x,y), POINT_HALO_RADIUS+2, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(img_copy, (x,y), POINT_HALO_RADIUS,   (0,0,0),       -1, cv2.LINE_AA)
        cv2.circle(img_copy, (x,y), POINT_RADIUS,        COLOR_POINT,   -1, cv2.LINE_AA)

        # ====== 放大過的點名（固定像素大小） ======
        # 先決定文字尺寸
        (tw, th), bl = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICK)

        # 位置（盡量不出界）
        lx = x + LABEL_OFFSET_X
        ly = y + LABEL_OFFSET_Y
        lx = min(max(0, lx), w - tw - 2*LABEL_PAD)
        ly = min(max(th + bl + LABEL_PAD, ly), h - LABEL_PAD)

        # 半透明底板區域
        x1, y1 = lx - LABEL_PAD, ly - th - bl - LABEL_PAD
        x2, y2 = lx + tw + LABEL_PAD, ly + LABEL_PAD
        draw_filled_box(img_copy, x1, y1, x2, y2, (0,0,0), alpha=LABEL_BG_ALPHA)

        # 黑描邊 + 白字（更清楚）
        cv2.putText(img_copy, lab, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, (0,0,0), LABEL_FONT_THICK+2, cv2.LINE_AA)
        cv2.putText(img_copy, lab, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, (255,255,255), LABEL_FONT_THICK,   cv2.LINE_AA)
        # =========================================

def draw_mouse_crosshair():
    cx, cy = mouse_xy
    cv2.drawMarker(img_copy, (cx,cy), (255,255,255), cv2.MARKER_CROSS, 18, 3)
    cv2.drawMarker(img_copy, (cx,cy), (0,0,0),       cv2.MARKER_CROSS, 16, 2)
    cv2.drawMarker(img_copy, (cx,cy), COLOR_CROSS,   cv2.MARKER_CROSS, 14, 1)

def render_zoom():
    """固定大小的 zoom 畫布 + 固定粗細十字與字級（不會跟倍率/半徑變）"""
    global SHOW_ZOOM
    if img is None or not SHOW_ZOOM: return

    cx, cy = mouse_xy; h, w = img.shape[:2]; r = zoom_radius
    # ROI (2r x 2r), clamp to edges
    x1 = clamp(cx - r, 0, w - 1); y1 = clamp(cy - r, 0, h - 1)
    x2 = clamp(cx + r, 0, w);     y2 = clamp(cy + r, 0, h)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return

    # scale by zoom_factor first (content magnification)
    scaled = cv2.resize(
        roi,
        (max(1, roi.shape[1] * zoom_factor), max(1, roi.shape[0] * zoom_factor)),
        interpolation=cv2.INTER_NEAREST
    )

    # letterbox into fixed canvas
    canvas = np.zeros((ZOOM_CANVAS, ZOOM_CANVAS, 3), dtype=scaled.dtype)
    sh, sw = scaled.shape[:2]
    scale  = min(ZOOM_CANVAS / sw, ZOOM_CANVAS / sh)
    new_w  = max(1, int(sw * scale)); new_h = max(1, int(sh * scale))
    view   = cv2.resize(scaled, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    ox     = (ZOOM_CANVAS - new_w) // 2; oy = (ZOOM_CANVAS - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = view

    # crosshair
    cv2.line(canvas, (ZOOM_CANVAS // 2, 0), (ZOOM_CANVAS // 2, ZOOM_CANVAS), COLOR_CROSS, CROSS_THICK, cv2.LINE_AA)
    cv2.line(canvas, (0, ZOOM_CANVAS // 2), (ZOOM_CANVAS, ZOOM_CANVAS // 2), COLOR_CROSS, CROSS_THICK, cv2.LINE_AA)
    cv2.circle(canvas, (ZOOM_CANVAS // 2, ZOOM_CANVAS // 2), 3, COLOR_CROSS, -1, cv2.LINE_AA)

    # adaptive label, never overflow
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_w = int(ZOOM_CANVAS * ZOOM_LABEL_MAX_FRAC)
    label_full = f"({cx},{cy})  z={zoom_factor}x  r={r}px"
    label_compact = f"z={zoom_factor}x  r={r}px"
    label = None; scale_txt = ZOOM_FONT_SCALE
    (tw, th), base = cv2.getTextSize(label_full, font, scale_txt, ZOOM_FONT_THICK)
    if tw <= max_w:
        label = label_full
    else:
        (tw, th), base = cv2.getTextSize(label_compact, font, scale_txt, ZOOM_FONT_THICK)
        if tw <= max_w:
            label = label_compact
        else:
            shrink = max(0.5, min(1.0, max_w / (tw + 1)))
            scale_txt = ZOOM_FONT_SCALE * shrink
            label = label_compact
            (tw, th), base = cv2.getTextSize(label, font, scale_txt, ZOOM_FONT_THICK)
    cv2.rectangle(canvas, (0, 0), (tw + 12, th + base + 8), COLOR_BG, -1)
    cv2.putText(canvas, label, (6, 6 + th), font, scale_txt, (220,220,220), ZOOM_FONT_THICK, cv2.LINE_AA)

    cv2.imshow(WIN_ZOOM, canvas)

# ------------- CSV / Image Save -------------
def save_csv(path=OUTPUT_CSV):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label","Image X","Image Y","Relative X","Relative Y","X (mm)","Y (mm)"])
        for p in measured_points:
            (ix, iy) = p['xy']; (rx, ry) = p['rel']
            (mx, my) = p['mm'] if p['mm'] is not None else (None, None)
            writer.writerow([p['label'], ix, iy, rx, ry, mx, my])
    print(f"[✓] Saved {len(measured_points)} rows to {os.path.abspath(path)}")
    # save annotated image
    try:
        update_display()
        out_img = img_copy.copy()
        img_name = os.path.splitext(path)[0] + ".png"
        cv2.imwrite(img_name, out_img)
        print(f"[✓] Saved annotated image to {os.path.abspath(img_name)}")
    except Exception as e:
        print(f"[!] Failed to save annotated image: {e}")

# ------------- Label generator -------------
def next_label():
    global label_index
    if label_index < len(labels):
        lab = labels[label_index]
    else:
        lab = f"P{label_index - len(labels) + 1}"
    label_index += 1
    return lab

# ------------- Mouse & Keys -------------
def on_mouse(event, x, y, flags, param):
    global mouse_xy, origin, x_axis_vec, y_axis_vec, pixel_to_mm_matrix
    global label_index, zoom_radius, mm_per_px_iso
    mouse_xy = (x, y)

    if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
        # Step 1: pick 4 axis midpoints (L, R, U, D)
        if len(axis_points) < 4:
            if _is_double_click(x, y):
                toast("Ignored double-click"); return

            axis_points.append((x, y))
            labels_lrud = ['Left', 'Right', 'Up', 'Down']
            print(f"[Axis] Picked {labels_lrud[len(axis_points)-1]} at ({x},{y})")

            if len(axis_points) == 2:
                lr_len = float(np.linalg.norm(np.array(axis_points[1]) - np.array(axis_points[0])))
                if lr_len < MIN_AXIS_LEN_PX:
                    axis_points.pop()
                    toast(f"Right 太靠近 Left（<{MIN_AXIS_LEN_PX}px）")
                    print(f"[Axis] Right too close to Left (<{MIN_AXIS_LEN_PX}px). Please pick Right again.")
                    update_display(); return
                v = np.array(axis_points[1]) - np.array(axis_points[0])
                x_axis_vec = unit(v)

            if len(axis_points) == 4:
                ud_len = float(np.linalg.norm(np.array(axis_points[2]) - np.array(axis_points[3])))
                if ud_len < MIN_AXIS_LEN_PX:
                    axis_points.pop()
                    toast(f"Down 太靠近 Up（<{MIN_AXIS_LEN_PX}px）")
                    print(f"[Axis] Down too close to Up (<{MIN_AXIS_LEN_PX}px). Please pick Down again.")
                    update_display(); return

                v = np.array(axis_points[2]) - np.array(axis_points[3])  # U - D：Y 向上為正
                y_axis_vec = unit(v)

                o = line_intersection(axis_points[0], axis_points[1], axis_points[2], axis_points[3])
                if o is None:
                    axis_points.pop()
                    toast("兩軸幾乎平行，請重選 Down")
                    print("[Axis] Lines are parallel or invalid. Please pick Down again.")
                    update_display(); return

                origin = o
                LR_len_px = float(np.linalg.norm(np.array(axis_points[1]) - np.array(axis_points[0])))
                UD_len_px = float(np.linalg.norm(np.array(axis_points[2]) - np.array(axis_points[3])))
                half_px_x = LR_len_px * 0.5; half_px_y = UD_len_px * 0.5
                half_mm_x = WIDTH_MM * 0.5;  half_mm_y = HEIGHT_MM * 0.5
                x_pt = ( int(round(origin[0] + x_axis_vec[0]*half_px_x)),
                         int(round(origin[1] + x_axis_vec[1]*half_px_x)) )
                y_pt = ( int(round(origin[0] + y_axis_vec[0]*half_px_y)),
                         int(round(origin[1] + y_axis_vec[1]*half_px_y)) )
                pixel_to_mm_matrix = build_pixel_to_mm(origin, x_pt, y_pt, half_mm_x, half_mm_y)

                mm_per_px_x = WIDTH_MM / LR_len_px if LR_len_px > 0 else float('nan')
                mm_per_px_y = HEIGHT_MM / UD_len_px if UD_len_px > 0 else float('nan')
                mm_per_px_iso = (mm_per_px_x + mm_per_px_y) / 2.0
                print(f"[Axis] Origin at {origin}. Calibrated to {WIDTH_MM}x{HEIGHT_MM} mm.")
                print(f"       Scale ≈ {mm_per_px_x:.6f} mm/px (X), {mm_per_px_y:.6f} mm/px (Y)")
                print("       Convention: X: Left -{:.0f} ↔ Right +{:.0f} (mm); Y: Down -{:.0f} ↔ Up +{:.0f} (mm)".format(
                    WIDTH_MM/2, WIDTH_MM/2, HEIGHT_MM/2, HEIGHT_MM/2
                ))

                # 若使用毫米指定半徑，完成校正後用實際比例精準換算一次
                if desired_radius_mm is not None and np.isfinite(mm_per_px_iso):
                    px_per_mm_iso = 1.0 / mm_per_px_iso if mm_per_px_iso > 0 else approx_px_per_mm
                    zoom_radius = clamp(int(desired_radius_mm * px_per_mm_iso), RADIUS_MIN_PX, RADIUS_MAX_PX)
                    print(f"       Zoom radius set to ~{desired_radius_mm} mm (≈{zoom_radius}px)")

            update_display()
            return

        # Step 2: measure points
        if origin is None or x_axis_vec is None or y_axis_vec is None:
            toast("軸線尚未完成（需 L,R,U,D 四點）")
            print("[Warn] Axis not ready. Pick 4 axis points first.")
            return

        rel_vec = np.array([x, y]) - np.array(origin)
        rel_x = float(np.dot(rel_vec, x_axis_vec))
        rel_y = float(np.dot(rel_vec, y_axis_vec))
        mmx, mmy = pixel_to_mm((x, y), pixel_to_mm_matrix)

        lab = next_label()
        measured_points.append({'label': lab, 'xy': (x, y), 'rel': (rel_x, rel_y), 'mm': (mmx, mmy)})
        print(f"[+] Pick: {lab} | image=({x},{y}) rel=({rel_x:.3f},{rel_y:.3f}) mm=({fmt(mmx)},{fmt(mmy)})")
        update_display()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if measured_points:
            removed = measured_points.pop()
            global label_index
            label_index = max(0, label_index - 1)
            print(f"[-] Undo: {removed['label']} at {removed['xy']}")
            update_display()

def on_mouse_zoom(event, x, y, flags, param):
    if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
        toast("請在主視窗（Image）點選座標")

def on_key(key):
    global zoom_factor, zoom_radius, SHOW_ZOOM, label_index
    if key == 27:   # ESC
        return False
    elif key == ord('z'):
        zoom_factor = clamp(zoom_factor + 1, 2, 12)
    elif key == ord('x'):
        zoom_factor = clamp(zoom_factor - 1, 2, 12)
    elif key == ord('['):
        zoom_radius = clamp(zoom_radius - 10, RADIUS_MIN_PX, RADIUS_MAX_PX)
    elif key == ord(']'):
        zoom_radius = clamp(zoom_radius + 10, RADIUS_MIN_PX, RADIUS_MAX_PX)
    elif key == ord('m'):
        SHOW_ZOOM = not SHOW_ZOOM
        if SHOW_ZOOM:
            try:
                cv2.namedWindow(WIN_ZOOM, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN_ZOOM, ZOOM_CANVAS, ZOOM_CANVAS)
            except Exception:
                pass
        else:
            try:
                cv2.destroyWindow(WIN_ZOOM)
            except Exception:
                pass
    elif key == ord('u'):
        if measured_points:
            removed = measured_points.pop()
            label_index = max(0, label_index - 1)
            print(f"[-] Undo: {removed['label']} at {removed['xy']}")
    elif key == ord('c'):
        measured_points.clear()
        label_index = 0
        print("[!] Cleared all measured points. Labels reset.")
    elif key == ord('r'):
        reset_all()
        print("[!] Reset axes. Please pick 4 midpoints (Left, Right, Up, Down).")
    elif key == ord('s'):
        save_csv(OUTPUT_CSV)
    return True

def reset_all():
    axis_points.clear()
    measured_points.clear()
    reset_calibration()
    global label_index
    label_index = 0

def reset_calibration():
    global origin, x_axis_vec, y_axis_vec, pixel_to_mm_matrix, mm_per_px_iso
    origin = None; x_axis_vec = None; y_axis_vec = None
    pixel_to_mm_matrix = None; mm_per_px_iso = None

# ------------- Display Refresh -------------
def update_display():
    global img_copy
    img_copy = img.copy()
    draw_axes_and_origin()
    draw_points_labels()
    draw_mouse_crosshair()

    # 狀態面板（多行）
    step = "Step 1: Pick axes L→R→U→D" if len(axis_points) < 4 else "Step 2: Left-click add point, Right-click undo"
    if mm_per_px_iso and np.isfinite(mm_per_px_iso):
        radius_mm_str = f"{zoom_radius * mm_per_px_iso:.2f} mm"
    else:
        radius_mm_str = f"{(zoom_radius / approx_px_per_mm):.2f} mm"
    lines = [
        step,
        f"Axes picked: {min(len(axis_points),4)}/4   Points: {len(measured_points)}",
        f"Zoom: {zoom_factor}x   Radius: {zoom_radius}px (~{radius_mm_str})   (m: toggle magnifier)",
        "Hotkeys: z/x zoom, [/] radius, u undo, c clear, r reset, s save, Esc exit"
    ]
    if time.time() < toast_deadline:
        lines.append(toast_msg)
    draw_status_box(lines)

    cv2.imshow(WIN_MAIN, img_copy)
    render_zoom()

# ------------- Main -------------
def main():
    global img, img_copy, IMG_PATH, OUTPUT_CSV, zoom_factor, zoom_radius, SHOW_ZOOM
    global desired_radius_mm, approx_px_per_mm, STATUS_FONT_SCALE

    args = parse_args()
    if args.image:  IMG_PATH = args.image
    if args.out:    OUTPUT_CSV = args.out
    if args.zoom is not None:   zoom_factor = clamp(int(args.zoom), 2, 12)
    if args.radius is not None: zoom_radius = clamp(int(args.radius), RADIUS_MIN_PX, RADIUS_MAX_PX)
    if args.radius_mm is not None:
        desired_radius_mm = max(0.1, float(args.radius_mm))
    elif INIT_RADIUS_MM is not None:
        desired_radius_mm = INIT_RADIUS_MM
    if args.no_zoom: SHOW_ZOOM = False

    print("=== Controls ===")
    print("  L-Click: pick axis (first 4) then measure points with labels (A,B,C,...)")
    print("  R-Click: undo last measured point")
    print("  z/x    : zoom +/-  |  [/]: radius +/-  |  m: toggle magnifier")
    print("  u/c    : undo / clear measured points  |  r: reset axes")
    print("  s      : save CSV + annotated PNG      |  Esc: exit (auto-save)")
    print("================\n")
    print(f"[Info] Physical size set to {WIDTH_MM} x {HEIGHT_MM} mm; origin at image center is (0,0) mm.")
    print("[Info] On-image shows *label only*. Full details are printed here and saved to CSV.")

    print(f"[Info] Loading image from: {os.path.abspath(IMG_PATH)}")
    if not os.path.exists(IMG_PATH):
        print("[Error] File not found. Tips:")
        print("  - Use an absolute path, or run with:  python image.py --image \"/full/path/to/your.jpg\"")
        print("  - If the path has spaces, wrap it in quotes")
        raise FileNotFoundError(f"File does not exist: {IMG_PATH}")
    img_bgr = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print("[Error] OpenCV failed to decode the file. Possible reasons: unsupported format or path encoding.")
        raise FileNotFoundError(f"Cannot read image (decode failed): {IMG_PATH}")

    img_bgr = img_bgr.copy()
    H, W = img_bgr.shape[:2]
    # 載圖後先用影像尺寸粗估 px/mm
    px_per_mm_x_approx = W / WIDTH_MM
    px_per_mm_y_approx = H / HEIGHT_MM
    approx_px_per_mm = (px_per_mm_x_approx + px_per_mm_y_approx) / 2.0

    # 若用毫米指定半徑，載圖後先粗估一個像素半徑
    if desired_radius_mm is not None:
        zoom_radius = clamp(int(desired_radius_mm * approx_px_per_mm), RADIUS_MIN_PX, RADIUS_MAX_PX)

    # 設定 HUD 字級（依固定像素高度，一次性計算）
    STATUS_FONT_SCALE = font_scale_for_height(HUD_TEXT_PX, STATUS_FONT_THICK)

    print(f"[Info] Expected extents: X in [-{WIDTH_MM/2:.0f}, +{WIDTH_MM/2:.0f}] mm; Y in [-{HEIGHT_MM/2:.0f}, +{HEIGHT_MM/2:.0f}] mm.")

    global LABEL_FONT_SCALE
    LABEL_FONT_SCALE = font_scale_for_height(LABEL_TEXT_PX, LABEL_FONT_THICK)


    # init windows
    global img, img_copy
    img = img_bgr; img_copy = img.copy()
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_ZOOM, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_ZOOM, ZOOM_CANVAS, ZOOM_CANVAS)
    cv2.setMouseCallback(WIN_MAIN, on_mouse)
    cv2.setMouseCallback(WIN_ZOOM, on_mouse_zoom)

    update_display()
    while True:
        key = cv2.waitKey(16) & 0xFF
        if not on_key(key): break
        update_display()

    cv2.destroyAllWindows()
    save_csv(OUTPUT_CSV)

if __name__ == "__main__":
    main()
