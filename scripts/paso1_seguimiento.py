"""Paso 1a: detección (RF-DETR) + seguimiento (Deep-EIoU propio + OSNet) sobre un tramo del clip.

Uso:
  python scripts/paso1_seguimiento.py VIDEO INICIO FIN SALIDA_DIR OSNET_PESOS

Salidas en SALIDA_DIR:
  tracks.json        trayectorias interpoladas por imagen
  balls.json         detecciones de balón por imagen
  ids_<n>.jpg        imágenes con cada jugador numerado (para elegir al 9)
  stats.json         tiempos y conteos
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
import torch
from rfdetr import RFDETRLarge
from torchreid.reid.utils import FeatureExtractor

from peak.tracking.deep_eiou import DeepEIoUTracker, interpolate

PERSON, BALL = 1, 37  # ids COCO en RF-DETR (rfdetr/assets/coco_classes.py)


def crops(img_rgb, boxes):
    h, w = img_rgb.shape[:2]
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        c = img_rgb[y1:max(y2, y1 + 2), x1:max(x2, x1 + 2)]
        out.append(c)
    return out


def main(video, start, end, outdir, osnet_path):
    os.makedirs(outdir, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    det = RFDETRLarge(device="cpu")
    reid = FeatureExtractor(model_name="osnet_x1_0", model_path=osnet_path, device="cpu", verbose=False)
    tracker = DeepEIoUTracker()
    balls, frames_bgr = {}, {}
    t_det = t_reid = t_trk = 0.0
    for f in range(start, end + 1):
        ok, bgr = cap.read()
        if not ok:
            break
        frames_bgr[f] = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        d = det.predict(rgb, threshold=0.1)
        t_det += time.perf_counter() - t0
        cls = d.class_id if d.class_id is not None else np.array([])
        p = d[cls == PERSON]
        b = d[cls == BALL]
        balls[f] = [[*map(float, bb), float(s)] for bb, s in zip(b.xyxy, b.confidence)]
        t0 = time.perf_counter()
        # Apariencia solo para confianza alta: es lo único que usa la etapa 1 de Deep-EIoU
        feats = np.full((len(p), 512), np.nan)
        hi = np.where(p.confidence > 0.6)[0]
        if len(hi):
            feats[hi] = reid(crops(rgb, p.xyxy[hi])).cpu().numpy()
        t_reid += time.perf_counter() - t0
        t0 = time.perf_counter()
        tracker.update(p.xyxy, p.confidence, feats, f)
        t_trk += time.perf_counter() - t0
    n = len(frames_bgr)

    tracks = {}
    for t in tracker.all_tracks():
        tracks[t.track_id] = {str(fr): [*map(float, bx), float(sc), interp]
                              for fr, (bx, sc, interp) in interpolate(t.history).items()}
    json.dump(tracks, open(os.path.join(outdir, "tracks.json"), "w"))
    json.dump(balls, open(os.path.join(outdir, "balls.json"), "w"))

    # Imágenes con IDs (ampliadas ×2 para que se lean)
    picks = [start, start + (n - 1) // 2, start + n - 1]
    for i, f in enumerate(picks):
        img = cv2.resize(frames_bgr[f], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for tid, tr in tracks.items():
            if str(f) in tr:
                x1, y1, x2, y2 = [int(v * 2) for v in tr[str(f)][:4]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img, str(tid), (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(img, str(tid), (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img, f"imagen {f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(outdir, f"ids_{i}.jpg"), img)

    stats = dict(frames=n, tracks=len(tracks), seg_deteccion_por_imagen=t_det / n,
                 seg_reid_por_imagen=t_reid / n, seg_tracker_por_imagen=t_trk / n)
    json.dump(stats, open(os.path.join(outdir, "stats.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
