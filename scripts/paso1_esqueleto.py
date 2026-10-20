"""Paso 1b: esqueleto del jugador elegido (ViTPose++ Huge) y video de salida.

Uso:
  python scripts/paso1_esqueleto.py VIDEO SEGUIMIENTO_DIR OBJETIVO SALIDA_MP4

OBJETIVO: identidad del 9, o tramos si el seguimiento le cambió el número,
  por ejemplo "16@0-795,21@796-99999" (identidad@imagen_inicio-imagen_fin).

Lee tracks.json y balls.json de SEGUIMIENTO_DIR. Escribe:
  SALIDA_MP4                 video ampliado ×2: todos los jugadores numerados, balón y esqueleto del 9
  pose_<ID>.json (en SEGUIMIENTO_DIR)  17 puntos COCO por imagen: x, y, confianza
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

MODEL = "usyd-community/vitpose-plus-huge"  # Apache 2.0 (ficha del modelo)
COCO_DATASET_INDEX = 0                        # ViTPose++ es multi-dataset; 0 = COCO (17 puntos)
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
            (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
KP_MIN_CONF = 0.3
SCALE = 2


def parse_target(spec, tracks):
    """Devuelve {imagen: caja} del 9 y el conjunto de identidades usadas."""
    target, ids = {}, set()
    for part in spec.split(","):
        if "@" in part:
            tid, rng = part.split("@")
            a, b = map(int, rng.split("-"))
        else:
            tid, a, b = part, -1, 10**9
        ids.add(tid)
        for f, v in tracks[tid].items():
            if a <= int(f) <= b:
                target[f] = v
    return target, ids


def main(video, trk_dir, target_spec, out_mp4):
    torch.set_num_threads(os.cpu_count())
    tracks = json.load(open(os.path.join(trk_dir, "tracks.json")))
    balls = json.load(open(os.path.join(trk_dir, "balls.json")))
    target, target_ids = parse_target(target_spec, tracks)
    target_id = "9"
    frames = sorted(int(f) for f in balls)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = VitPoseForPoseEstimation.from_pretrained(MODEL).eval()

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * SCALE, h * SCALE))
    poses, t_pose = {}, []
    for f in frames:
        ok, bgr = cap.read()
        if not ok:
            break
        img = cv2.resize(bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        tbox = target.get(str(f))
        for tid, tr in tracks.items():
            if str(f) in tr:
                if tbox is not None and tr[str(f)][:4] == tbox[:4]:
                    continue  # el 9 se dibuja aparte
                x1, y1, x2, y2 = [int(v * SCALE) for v in tr[str(f)][:4]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(img, tid, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        interpolated = tbox is not None and bool(tbox[5])
        if tbox is not None:
            x1, y1, x2, y2 = [int(v * SCALE) for v in tbox[:4]]
            # Caja interpolada = el detector no vio al 9 en esta imagen; la posición es estimada
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1 if interpolated else 2)
            cv2.putText(img, "9 (estimado)" if interpolated else "9", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for bx in balls.get(str(f), []):
            if bx[4] >= 0.3:
                cx, cy = int((bx[0] + bx[2]) / 2 * SCALE), int((bx[1] + bx[3]) / 2 * SCALE)
                cv2.circle(img, (cx, cy), 8, (255, 255, 255), 2)
        # Sin esqueleto en imágenes interpoladas: la caja no está confirmada por el detector
        if str(f) in target and not interpolated:
            x1, y1, x2, y2 = target[str(f)][:4]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            box = [[[x1, y1, x2 - x1, y2 - y1]]]  # formato COCO: x, y, ancho, alto
            t0 = time.perf_counter()
            inputs = proc(rgb, boxes=box, return_tensors="pt")
            with torch.inference_mode():
                out = model(**inputs, dataset_index=torch.tensor([COCO_DATASET_INDEX]))
            res = proc.post_process_pose_estimation(out, boxes=box)[0][0]
            t_pose.append(time.perf_counter() - t0)
            kps = res["keypoints"].numpy()
            sc = res["scores"].numpy()
            poses[f] = [[float(x), float(y), float(s)] for (x, y), s in zip(kps, sc)]
            for a, b in SKELETON:
                if sc[a] >= KP_MIN_CONF and sc[b] >= KP_MIN_CONF:
                    pa = tuple(int(v * SCALE) for v in kps[a])
                    pb = tuple(int(v * SCALE) for v in kps[b])
                    cv2.line(img, pa, pb, (0, 255, 0), 2)
            for (x, y), s in zip(kps, sc):
                if s >= KP_MIN_CONF:
                    cv2.circle(img, (int(x * SCALE), int(y * SCALE)), 3, (0, 0, 255), -1)
        cv2.putText(img, f"imagen {f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(img)
    vw.release()
    json.dump(poses, open(os.path.join(trk_dir, f"pose_{target_id}.json"), "w"))
    n_interp = sum(1 for v in target.values() if v[5])
    stats = dict(imagenes=len(frames), imagenes_con_9=len(target), imagenes_9_interpoladas=n_interp,
                 imagenes_con_esqueleto=len(poses),
                 seg_pose_por_imagen=float(np.mean(t_pose)) if t_pose else None)
    json.dump(stats, open(os.path.join(trk_dir, f"pose_{target_id}_stats.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
