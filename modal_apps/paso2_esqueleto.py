"""Esqueleto del 9 (ViTPose++ Huge) en Modal, GPU T4, sobre el clip del paso 2.

La inferencia corre en Modal; el video de salida se dibuja localmente con
la misma lógica del paso 1 (scripts/paso1_esqueleto.py).

Uso (desde la raíz del repositorio):
  modal run modal_apps/paso2_esqueleto.py --video CLIP.mp4 --resultados resultados/paso2_vanpersie \
      --nueve "24@83-238" --salida-mp4 resultados/paso2_vanpersie/esqueleto_9.mp4

Tope de costo: timeout de 300 s en T4.
"""
import json
import os

import modal

MODEL = "usyd-community/vitpose-plus-huge"  # Apache 2.0, el mismo del paso 1
COCO_DATASET_INDEX = 0
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
            (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
KP_MIN_CONF = 0.3
NOMBRES = ["nariz", "ojo_izq", "ojo_der", "oreja_izq", "oreja_der", "hombro_izq", "hombro_der",
           "codo_izq", "codo_der", "muñeca_izq", "muñeca_der", "cadera_izq", "cadera_der",
           "rodilla_izq", "rodilla_der", "tobillo_izq", "tobillo_der"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers==5.20.2", "opencv-python-headless", "numpy", "pillow", "scipy")
)
app = modal.App("peak-paso2-esqueleto", image=image)
vol = modal.Volume.from_name("peak-resultados", create_if_missing=True)


@app.function(gpu="T4", cpu=2.0, memory=16384, timeout=300, volumes={"/resultados": vol})
def esqueleto(video_bytes: bytes, cajas: dict, etiqueta: str) -> dict:
    """cajas: {imagen: [x1, y1, x2, y2]} solo de imágenes donde el detector vio al 9."""
    import tempfile
    import time
    import cv2
    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
        fh.write(video_bytes)
        ruta = fh.name
    proc = AutoProcessor.from_pretrained(MODEL)
    model = VitPoseForPoseEstimation.from_pretrained(MODEL).to("cuda").eval()
    cap = cv2.VideoCapture(ruta)
    ultima = max(int(f) for f in cajas)
    poses, tiempos = {}, []
    for f in range(ultima + 1):
        ok, bgr = cap.read()
        if not ok:
            break
        if str(f) not in cajas:
            continue
        x1, y1, x2, y2 = cajas[str(f)]
        box = [[[x1, y1, x2 - x1, y2 - y1]]]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        inputs = proc(rgb, boxes=box, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model(**inputs, dataset_index=torch.tensor([COCO_DATASET_INDEX], device="cuda"))
        res = proc.post_process_pose_estimation(out, boxes=box)[0][0]
        tiempos.append(time.perf_counter() - t0)
        kps = res["keypoints"].cpu().numpy()
        sc = res["scores"].cpu().numpy()
        poses[str(f)] = [[float(x), float(y), float(s)] for (x, y), s in zip(kps, sc)]
    out = {"poses": poses, "seg_por_imagen": tiempos, "gpu": torch.cuda.get_device_name(0)}
    os.makedirs(f"/resultados/{etiqueta}", exist_ok=True)
    json.dump(out, open(f"/resultados/{etiqueta}/pose_9.json", "w"))
    vol.commit()
    return out


@app.local_entrypoint()
def main(video: str, resultados: str, nueve: str, salida_mp4: str):
    import cv2
    import numpy as np

    tracks = json.load(open(os.path.join(resultados, "tracks.json")))
    balls = json.load(open(os.path.join(resultados, "balls.json")))
    objetivo = {}
    for parte in nueve.split(","):
        tid, rng = parte.split("@")
        a, b = map(int, rng.split("-"))
        for f, v in tracks[tid].items():
            if a <= int(f) <= b:
                objetivo[f] = v
    # Sin esqueleto en imágenes interpoladas (el detector no vio al 9)
    cajas = {f: v[:4] for f, v in objetivo.items() if not v[5]}
    r = esqueleto.remote(open(video, "rb").read(), cajas, os.path.basename(os.path.normpath(resultados)))
    poses = r["poses"]
    json.dump(poses, open(os.path.join(resultados, "pose_9.json"), "w"))

    # Video de salida: igual que el paso 1 (ampliado ×2, todos numerados, esqueleto del 9)
    S = 2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    frames = sorted(int(f) for f in balls)
    vw = cv2.VideoWriter(salida_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * S, h * S))
    for f in range(frames[-1] + 1):
        ok, bgr = cap.read()
        if not ok:
            break
        if f < frames[0]:
            continue
        img = cv2.resize(bgr, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
        tbox = objetivo.get(str(f))
        for tid, tr in tracks.items():
            if str(f) in tr and not (tbox is not None and tr[str(f)][:4] == tbox[:4]):
                x1, y1, x2, y2 = [int(v * S) for v in tr[str(f)][:4]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(img, tid, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if tbox is not None:
            interp = bool(tbox[5])
            x1, y1, x2, y2 = [int(v * S) for v in tbox[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1 if interp else 2)
            cv2.putText(img, "9 (estimado)" if interp else "9", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for bx in balls.get(str(f), []):
            if bx[4] >= 0.3:
                cv2.circle(img, (int((bx[0] + bx[2]) / 2 * S), int((bx[1] + bx[3]) / 2 * S)), 8, (255, 255, 255), 2)
        p = poses.get(str(f))
        if p is not None:
            kps = np.array([[x, y] for x, y, _ in p])
            sc = np.array([s for _, _, s in p])
            for a, b in SKELETON:
                if sc[a] >= KP_MIN_CONF and sc[b] >= KP_MIN_CONF:
                    cv2.line(img, tuple(int(v * S) for v in kps[a]), tuple(int(v * S) for v in kps[b]), (0, 255, 0), 2)
            for (x, y), s in zip(kps, sc):
                if s >= KP_MIN_CONF:
                    cv2.circle(img, (int(x * S), int(y * S)), 3, (0, 0, 255), -1)
        cv2.putText(img, f"imagen {f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(img)
    vw.release()

    sc = np.array([[s for _, _, s in p] for p in poses.values()])
    alturas = [objetivo[f][3] - objetivo[f][1] for f in poses]
    stats = {
        "imagenes_con_9": len(objetivo),
        "imagenes_9_interpoladas": sum(1 for v in objetivo.values() if v[5]),
        "imagenes_con_esqueleto": len(poses),
        "pct_puntos_sobre_0_3": float(100 * (sc >= KP_MIN_CONF).mean()),
        "pct_por_punto": {n: float(100 * (sc[:, i] >= KP_MIN_CONF).mean()) for i, n in enumerate(NOMBRES)},
        "altura_caja_9_px": {"min": float(min(alturas)), "mediana": float(np.median(alturas)), "max": float(max(alturas))},
        "seg_pose_por_imagen_gpu": float(np.mean(r["seg_por_imagen"])),
        "gpu": r["gpu"],
    }
    json.dump(stats, open(os.path.join(resultados, "pose_9_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1, ensure_ascii=False))
