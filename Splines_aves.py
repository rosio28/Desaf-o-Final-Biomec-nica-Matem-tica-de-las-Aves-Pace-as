# ============================================================
#  PROYECTO FINAL: SPLINES + IA + FOURIER
#  Colibrí gigante (Patagona gigas)
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from scipy.fft import rfft, rfftfreq

# IA ligera para detección del ave
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================
VIDEO_PATH = "Patagona gigas.mp4"      #video
OUT_DIR = "salida_proyecto"
MODEL_NAME = "yolov8n.pt"         # se descarga automáticamente

MAX_FEATURES = 120
MIN_FEATURE_QUALITY = 0.01
MIN_FEATURE_DISTANCE = 5
REDETECT_IF_LOST = False         


# ============================================================
# UTILIDADES
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def expand_bbox(bbox, frame_w, frame_h, margin=0.15):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    x1 = int(x1 - margin * bw)
    y1 = int(y1 - margin * bh)
    x2 = int(x2 + margin * bw)
    y2 = int(y2 + margin * bh)
    return clamp_bbox(x1, y1, x2, y2, frame_w, frame_h)


def draw_box(frame, bbox, color=(0, 255, 0), thickness=2):
    if bbox is None:
        return frame
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, "Bird ROI", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame


def put_text(frame, text, org, scale=0.6, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)
    return frame


def read_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()

    if len(frames) < 3:
        raise RuntimeError("El video tiene muy pocos frames.")

    return frames, fps


def detect_bird_bbox(frame, model, conf=0.25):
    """
    Busca la clase 'bird' en COCO.
    Devuelve bbox (x1,y1,x2,y2) o None.
    """
    h, w = frame.shape[:2]
    result = model.predict(frame, conf=conf, verbose=False)[0]

    best = None
    best_conf = -1.0

    if result.boxes is None:
        return None

    for box in result.boxes:
        cls_id = int(box.cls.item())
        name = model.names[cls_id]

        if name != "bird":
            continue

        score = float(box.conf.item())
        if score > best_conf:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            best = (x1, y1, x2, y2)
            best_conf = score

    if best is None:
        return None

    return expand_bbox(best, w, h, margin=0.15)


def init_features(gray, bbox):
    """
    Detecta puntos de interés dentro del bbox usando Shi-Tomasi.
    """
    mask = np.zeros_like(gray)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 255

    p0 = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        maxCorners=MAX_FEATURES,
        qualityLevel=MIN_FEATURE_QUALITY,
        minDistance=MIN_FEATURE_DISTANCE,
        blockSize=7
    )
    return p0


def manual_click_point(frame, bbox=None):
    """
    Fallback manual: abre ventana y permite hacer click en un punto.
    Se usa solo si la IA/feature tracking no encuentra nada útil.
    """
    temp = frame.copy()
    if bbox is not None:
        temp = draw_box(temp, bbox, color=(0, 255, 255), thickness=2)

    clicked = {"pt": None}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (x, y)

    win = "Haz click en la punta del ala y presiona ENTER"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_callback)

    while True:
        show = temp.copy()
        if clicked["pt"] is not None:
            cv2.circle(show, clicked["pt"], 6, (0, 0, 255), -1)
            put_text(show, "Punto elegido", (20, 40), scale=0.9, color=(0, 0, 255), thickness=2)

        cv2.imshow(win, show)
        key = cv2.waitKey(20) & 0xFF

        if key == 13 or key == 10:  # ENTER
            break
        if key == 27:  # ESC
            break

    cv2.destroyWindow(win)

    if clicked["pt"] is None:
        raise RuntimeError("No se seleccionó un punto manualmente.")

    x, y = clicked["pt"]
    return np.array([[[float(x), float(y)]]], dtype=np.float32)


def fill_nans_1d(v):
    """
    Rellena valores NaN por interpolación lineal.
    """
    v = np.asarray(v, dtype=float)
    idx = np.arange(len(v))
    m = np.isfinite(v)
    if m.sum() < 2:
        raise RuntimeError("No hay suficientes valores válidos para interpolar.")
    return np.interp(idx, idx[m], v[m])


def safe_nanstd(arr):
    arr = np.asarray(arr, dtype=float)
    if np.isfinite(arr).sum() == 0:
        return 0.0
    return float(np.nanstd(arr))


# ============================================================
# TRACKING
# ============================================================
def track_points(frames, fps, model):
    """
    1) Detecta el ave en el primer frame.
    2) Extrae puntos de interés dentro del ave.
    3) Los sigue con Lucas-Kanade.
    4) Elige el punto con mayor variación vertical como "ala".
    """
    n_frames = len(frames)
    first = frames[0]
    gray_prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    h, w = gray_prev.shape[:2]

    bbox = detect_bird_bbox(first, model)
    if bbox is None:
        # Si no detecta el ave, usa todo el frame como región.
        bbox = (0, 0, w - 1, h - 1)

    p0 = init_features(gray_prev, bbox)

    if p0 is None or len(p0) == 0:
        # Fallback manual: un clic en la punta del ala.
        p0 = manual_click_point(first, bbox=bbox)

    p0 = p0.reshape(-1, 1, 2).astype(np.float32)
    n_points = len(p0)

    # track_records[point_id, frame_id, coord]
    track_records = np.full((n_points, n_frames, 2), np.nan, dtype=np.float32)

    # guardar frame 0
    for pid in range(n_points):
        track_records[pid, 0, 0] = p0[pid, 0, 0]
        track_records[pid, 0, 1] = p0[pid, 0, 1]

    prev_pts = p0.copy()
    active_ids = list(range(n_points))

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # tracking frame a frame
    for i in range(1, n_frames):
        gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        if len(prev_pts) == 0:
            break

        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_curr, prev_pts, None, **lk_params
        )

        status = status.reshape(-1)

        new_pts = []
        new_ids = []

        for j, ok in enumerate(status):
            pid = active_ids[j]
            if ok == 1:
                x, y = next_pts[j, 0]
                track_records[pid, i, 0] = x
                track_records[pid, i, 1] = y
                new_pts.append([[x, y]])
                new_ids.append(pid)
            else:
                # el punto se perdió en este frame
                track_records[pid, i, 0] = np.nan
                track_records[pid, i, 1] = np.nan

        prev_pts = np.array(new_pts, dtype=np.float32).reshape(-1, 1, 2) if len(new_pts) else np.empty((0, 1, 2), dtype=np.float32)
        active_ids = new_ids
        gray_prev = gray_curr

    # escoger el punto que más se mueve verticalmente
    scores = []
    for pid in range(n_points):
        y = track_records[pid, :, 1]
        x = track_records[pid, :, 0]
        valid_count = np.isfinite(y).sum()
        if valid_count < max(8, n_frames // 5):
            scores.append(-1.0)
            continue

        y_std = safe_nanstd(y)
        x_std = safe_nanstd(x)
        # prioriza movimiento vertical, pero sin perder estabilidad
        score = 1.8 * y_std + 0.4 * x_std + 0.01 * valid_count
        scores.append(score)

    best_id = int(np.argmax(scores))
    best_track = track_records[best_id].copy()

    x_best = fill_nans_1d(best_track[:, 0])
    y_best = fill_nans_1d(best_track[:, 1])

    # tiempo por frame
    t = np.arange(n_frames, dtype=float) / float(fps)

    return {
        "bbox": bbox,
        "track_records": track_records,
        "best_id": best_id,
        "best_track_x": x_best,
        "best_track_y": y_best,
        "t": t,
        "fps": fps,
        "n_frames": n_frames
    }


# ============================================================
# SPLINES CÚBICOS Si​(x)=ai​+bi​(x−xi​)+ci​(x−xi​)2+di​(x−xi​)3
# ============================================================
def build_spline_family(t, v):
    """
    Construye:
      - natural
      - clamped
      - not-a-knot

    Fórmula general por tramo:
      S_i(u) = a_i + b_i*u + c_i*u^2 + d_i*u^3
      con u = t - t_i
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)

    dv0 = (v[1] - v[0]) / (t[1] - t[0])
    dvn = (v[-1] - v[-2]) / (t[-1] - t[-2])

    cs_natural = CubicSpline(t, v, bc_type="natural")
    cs_clamped = CubicSpline(t, v, bc_type=((1, dv0), (1, dvn)))
    cs_notaknot = CubicSpline(t, v, bc_type="not-a-knot")

    return cs_natural, cs_clamped, cs_notaknot


def dominant_frequency(signal, fs):
    """
    FFT de una señal real:
      X_k = sum x_n * exp(-i*2*pi*k*n/N)
    Se ignora la componente DC.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)

    N = len(signal)
    yf = np.abs(rfft(signal))
    xf = rfftfreq(N, d=1.0 / fs)

    # ignorar DC
    if len(yf) > 1:
        idx = np.argmax(yf[1:]) + 1
    else:
        idx = 0

    return xf, yf, float(xf[idx])


def make_dense_eval(t, cs, n=1200):
    td = np.linspace(t.min(), t.max(), n)
    return td, cs(td), cs(td, 1), cs(td, 2)


# ============================================================
# GRAFICACIÓN EXPLICATIVA
# ============================================================
def save_spline_plots(out_dir, t, x, y, t_dense, x_nat, y_nat, x_cl, y_cl, x_nak, y_nak, fps):
    # 1) Comparación de trayectorias
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, "o", markersize=4, label="Puntos tracking")
    plt.plot(x_nat, y_nat, linewidth=2.5, label="Spline natural")
    plt.plot(x_cl, y_cl, "--", linewidth=2, label="Spline clamped")
    plt.plot(x_nak, y_nak, ":", linewidth=2.5, label="Spline not-a-knot")
    plt.title("Interpolación con splines cúbicos")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_splines_comparacion.png"), dpi=200)
    plt.close()

    # 2) Posición, velocidad y aceleración de x(t) e y(t)
    csx_nat, _, _ = build_spline_family(t, x)
    csy_nat, _, _ = build_spline_family(t, y)

    td = np.linspace(t.min(), t.max(), 1200)

    x0 = csx_nat(td)
    x1 = csx_nat(td, 1)
    x2 = csx_nat(td, 2)

    y0 = csy_nat(td)
    y1 = csy_nat(td, 1)
    y2 = csy_nat(td, 2)

    fig, ax = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    fig.suptitle("Posición, velocidad y aceleración obtenidas del spline natural", fontsize=14)

    ax[0, 0].plot(td, x0)
    ax[0, 0].set_title("x(t)")
    ax[0, 0].grid(True, alpha=0.3)

    ax[1, 0].plot(td, x1)
    ax[1, 0].set_title("x'(t)  → continuidad C1")
    ax[1, 0].grid(True, alpha=0.3)

    ax[2, 0].plot(td, x2)
    ax[2, 0].set_title("x''(t) → continuidad C2")
    ax[2, 0].grid(True, alpha=0.3)

    ax[0, 1].plot(td, y0)
    ax[0, 1].set_title("y(t)")
    ax[0, 1].grid(True, alpha=0.3)

    ax[1, 1].plot(td, y1)
    ax[1, 1].set_title("y'(t)  → continuidad C1")
    ax[1, 1].grid(True, alpha=0.3)

    ax[2, 1].plot(td, y2)
    ax[2, 1].set_title("y''(t) → continuidad C2")
    ax[2, 1].grid(True, alpha=0.3)

    for i in range(3):
        ax[i, 0].set_ylabel("valor")
        ax[i, 1].set_ylabel("valor")
    ax[2, 0].set_xlabel("tiempo (s)")
    ax[2, 1].set_xlabel("tiempo (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "02_derivadas_spline.png"), dpi=200)
    plt.close()

    # 3) FFT
    signal = y - np.mean(y)
    xf, yf, f0 = dominant_frequency(signal, fps)

    plt.figure(figsize=(10, 5))
    plt.plot(xf, yf)
    plt.title(f"Espectro de Fourier del ala | Frecuencia dominante ≈ {f0:.3f} Hz")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_fft.png"), dpi=200)
    plt.close()

    return f0


# ============================================================
# VIDEO ANOTADO
# ============================================================
def make_annotated_video(frames, out_path, bbox, track_x, track_y, fps):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # línea de trayectoria acumulada
    pts = []

    for i, frame in enumerate(frames):
        img = frame.copy()
        img = draw_box(img, bbox, color=(0, 255, 0), thickness=2)

        x = int(track_x[i])
        y = int(track_y[i])
        pts.append((x, y))

        # trayectoria
        for j in range(1, len(pts)):
            cv2.line(img, pts[j - 1], pts[j], (0, 255, 255), 2)

        # punto actual
        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

        put_text(img, f"Frame: {i+1}/{len(frames)}", (20, 35), scale=0.8, color=(255, 255, 255), thickness=2)
        put_text(img, "Tracking IA + Optical Flow", (20, 65), scale=0.8, color=(255, 255, 255), thickness=2)

        writer.write(img)

    writer.release()


# ============================================================
# REPORTE TEXTO
# ============================================================
def save_report(out_dir, fps, n_frames, best_id, f0, bbox):
    text = []
    text.append("REPORTE DEL PROYECTO")
    text.append("====================")
    text.append(f"FPS del video: {fps:.3f}")
    text.append(f"Frames totales: {n_frames}")
    text.append(f"Punto seleccionado: {best_id}")
    text.append(f"Frecuencia dominante estimada: {f0:.6f} Hz")
    text.append(f"BBox inicial: {bbox}")
    text.append("")
    text.append("Resumen matemático:")
    text.append("- Splines cúbicos por tramos.")
    text.append("- Continuidad C0, C1 y C2.")
    text.append("- Fourier sobre la señal temporal del ala.")
    text.append("- Comparación natural / clamped / not-a-knot.")

    with open(os.path.join(out_dir, "04_reporte.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(text))


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dir(OUT_DIR)

    print("Cargando video...")
    frames, fps = read_all_frames(VIDEO_PATH)
    print(f"Frames cargados: {len(frames)} | FPS: {fps:.3f}")

    print("Cargando modelo de IA...")
    model = YOLO(MODEL_NAME)

    print("Analizando y siguiendo puntos...")
    result = track_points(frames, fps, model)

    bbox = result["bbox"]
    t = result["t"]
    x = result["best_track_x"]
    y = result["best_track_y"]
    best_id = result["best_id"]

    # Splines para x(t) e y(t)
    csx_nat, csx_cl, csx_nak = build_spline_family(t, x)
    csy_nat, csy_cl, csy_nak = build_spline_family(t, y)

    t_dense = np.linspace(t.min(), t.max(), 1400)

    x_nat = csx_nat(t_dense)
    y_nat = csy_nat(t_dense)

    x_cl = csx_cl(t_dense)
    y_cl = csy_cl(t_dense)

    x_nak = csx_nak(t_dense)
    y_nak = csy_nak(t_dense)

    # Guardar gráficas explicativas
    print("Generando gráficas...")
    f0 = save_spline_plots(
        OUT_DIR, t, x, y, t_dense,
        x_nat, y_nat, x_cl, y_cl, x_nak, y_nak, fps
    )

    # Guardar video anotado
    print("Generando video anotado...")
    make_annotated_video(
        frames,
        os.path.join(OUT_DIR, "05_tracking_anotado.mp4"),
        bbox,
        x,
        y,
        fps
    )

    # Guardar datos crudos
    data = np.column_stack([t, x, y])
    np.savetxt(
        os.path.join(OUT_DIR, "06_puntos_tracking.csv"),
        data,
        delimiter=",",
        header="t,x,y",
        comments=""
    )

    # Guardar reporte
    save_report(OUT_DIR, fps, len(frames), best_id, f0, bbox)

    # Mostrar validación rápida en consola
    print("\n=== VALIDACIÓN ===")
    print(f"Punto elegido: {best_id}")
    print(f"Frecuencia dominante estimada: {f0:.3f} Hz")
    print("Archivos generados en la carpeta:", OUT_DIR)
    print(" - 01_splines_comparacion.png")
    print(" - 02_derivadas_spline.png")
    print(" - 03_fft.png")
    print(" - 04_reporte.txt")
    print(" - 05_tracking_anotado.mp4")
    print(" - 06_puntos_tracking.csv")

    # Gráfica final en pantalla
    plt.figure(figsize=(9, 6))
    plt.plot(x, y, "o", label="Tracking")
    plt.plot(x_nat, y_nat, linewidth=2.5, label="Spline natural")
    plt.title("Resultado final del spline sobre el punto seleccionado")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
