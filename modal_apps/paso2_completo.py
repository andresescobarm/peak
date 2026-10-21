"""Paso 2 completo en Modal: seguimiento (paso 1) + calibración de cancha (PnLCalib).

Dos funciones en GPU T4, cada una con su propio entorno:
- seguimiento: RF-DETR Large + Deep-EIoU propio + OSNet (MSMT17), igual que el paso 1.
- calibrar: PnLCalib (pesos SV, refinamiento PnL), igual que paso2_calibracion.py.

Tope de costo: timeouts de 500 s + 400 s = 900 s de T4 en total.

Uso (desde la raíz del repositorio):
  modal run modal_apps/paso2_completo.py --video CLIP.mp4 --inicio 0 --fin 238 --salida resultados/paso2_vanpersie
"""
import io
import json
import os

import modal

PNLCALIB_COMMIT = "8c87391d6f4ea40c5e4d65e61529916c7a49ce62"
PESOS_PNL = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0"
OSNET_URL = ("https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
             "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")

img_calib = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.3.1", "torchvision==0.18.1",
                 extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy==1.26.4", "opencv-python-headless==4.10.0.84", "scipy==1.13.1",
                 "shapely==2.0.7", "lsq-ellipse==2.2.1", "PyYAML==6.0.2",
                 "matplotlib==3.9.4", "tqdm==4.67.1")
    .run_commands(
        "git clone https://github.com/mguti97/PnLCalib /opt/PnLCalib",
        f"cd /opt/PnLCalib && git checkout {PNLCALIB_COMMIT}",
        "mkdir -p /opt/PnLCalib/weights",
        f"wget -q -O /opt/PnLCalib/weights/SV_kp {PESOS_PNL}/SV_kp",
        f"wget -q -O /opt/PnLCalib/weights/SV_lines {PESOS_PNL}/SV_lines",
    )
)

img_track = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("rfdetr==1.11.2", "torchreid==0.2.5", "tensorboard", "gdown",
                 "opencv-python-headless", "scipy", "numpy")
    .run_commands(f"mkdir -p /opt/w && wget -q -O /opt/w/osnet_x1_0_msmt17.pth '{OSNET_URL}'")
    .add_local_dir(os.path.join(os.path.dirname(__file__), "..", "src", "peak"), "/opt/src/peak")
)

app = modal.App("peak-paso2-completo")
vol = modal.Volume.from_name("peak-resultados", create_if_missing=True)

PERSON, BALL = 1, 37  # ids COCO en RF-DETR, como en el paso 1


def leer_tramo(video_bytes, inicio, fin):
    import tempfile
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
        fh.write(video_bytes)
        ruta = fh.name
    cap = cv2.VideoCapture(ruta)
    frames = []
    for i in range(fin + 1):
        ok, f = cap.read()
        if not ok:
            break
        if i >= inicio:
            frames.append(f)
    return frames


@app.function(image=img_track, gpu="T4", cpu=2.0, memory=16384, timeout=500, volumes={"/resultados": vol})
def seguimiento(video_bytes: bytes, inicio: int, fin: int, etiqueta: str) -> dict:
    import sys
    import time
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "/opt/src")
    import cv2
    import numpy as np
    import torch
    from rfdetr import RFDETRLarge
    from torchreid.reid.utils import FeatureExtractor
    from peak.tracking.deep_eiou import DeepEIoUTracker, interpolate

    frames = leer_tramo(video_bytes, inicio, fin)
    det = RFDETRLarge(device="cuda")
    reid = FeatureExtractor(model_name="osnet_x1_0", model_path="/opt/w/osnet_x1_0_msmt17.pth",
                            device="cuda", verbose=False)
    tracker = DeepEIoUTracker()
    balls = {}
    t_det = t_reid = 0.0
    for k, bgr in enumerate(frames):
        f = inicio + k
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        d = det.predict(rgb, threshold=0.1)
        t_det += time.perf_counter() - t0
        cls = d.class_id if d.class_id is not None else np.array([])
        p = d[cls == PERSON]
        b = d[cls == BALL]
        balls[f] = [[*map(float, bb), float(s)] for bb, s in zip(b.xyxy, b.confidence)]
        t0 = time.perf_counter()
        feats = np.full((len(p), 512), np.nan)
        hi = np.where(p.confidence > 0.6)[0]
        if len(hi):
            h, w = rgb.shape[:2]
            crops = []
            for x1, y1, x2, y2 in p.xyxy[hi]:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
                crops.append(rgb[y1:max(y2, y1 + 2), x1:max(x2, x1 + 2)])
            feats[hi] = reid(crops).cpu().numpy()
        t_reid += time.perf_counter() - t0
        tracker.update(p.xyxy, p.confidence, feats, f)

    tracks = {}
    for t in tracker.all_tracks():
        tracks[t.track_id] = {str(fr): [*map(float, bx), float(sc), interp]
                              for fr, (bx, sc, interp) in interpolate(t.history).items()}
    n = len(frames)
    out = {"tracks": tracks, "balls": balls,
           "stats": {"imagenes": n, "trayectorias": len(tracks),
                     "seg_deteccion_por_imagen": t_det / n, "seg_reid_por_imagen": t_reid / n,
                     "gpu": torch.cuda.get_device_name(0)}}
    os.makedirs(f"/resultados/{etiqueta}", exist_ok=True)
    json.dump(out, open(f"/resultados/{etiqueta}/seguimiento.json", "w"))
    vol.commit()
    return out


@app.function(image=img_calib, gpu="T4", cpu=2.0, memory=16384, timeout=400, volumes={"/resultados": vol})
def calibrar(video_bytes: bytes, inicio: int, fin: int, etiqueta: str) -> dict:
    import sys
    import time
    import torch
    import yaml
    os.chdir("/opt/PnLCalib")
    sys.path.insert(0, "/opt/PnLCalib")
    import inference as inf
    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l
    from utils.utils_calib import FramebyFrameCalib
    import torchvision.transforms as T

    frames = leer_tramo(video_bytes, inicio, fin)
    h, w = frames[0].shape[:2]
    device = "cuda:0"
    model = get_cls_net(yaml.safe_load(open("config/hrnetv2_w48.yaml")))
    model.load_state_dict(torch.load("weights/SV_kp", map_location=device))
    model.to(device).eval()
    model_l = get_cls_net_l(yaml.safe_load(open("config/hrnetv2_w48_l.yaml")))
    model_l.load_state_dict(torch.load("weights/SV_lines", map_location=device))
    model_l.to(device).eval()
    inf.device = device
    inf.transform2 = T.Resize((540, 960))

    params, tiempos = [], []
    for fr in frames:
        cam = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
        t1 = time.perf_counter()
        p = inf.inference(cam, fr.copy(), model, model_l, 0.3434, 0.7867, True)
        tiempos.append(time.perf_counter() - t1)
        if p is not None:
            P = inf.projection_from_cam_params(p)
            params.append({"cam_params": p["cam_params"], "P": P.tolist()})
        else:
            params.append(None)
    out = {"inicio": inicio, "params": params, "seg_por_imagen": tiempos,
           "gpu": torch.cuda.get_device_name(0), "ancho": w, "alto": h}
    os.makedirs(f"/resultados/{etiqueta}", exist_ok=True)
    json.dump(out, open(f"/resultados/{etiqueta}/calibracion.json", "w"))
    vol.commit()
    return out


@app.local_entrypoint()
def main(video: str, inicio: int, fin: int, salida: str):
    os.makedirs(salida, exist_ok=True)
    etiqueta = os.path.basename(os.path.normpath(salida))
    data = open(video, "rb").read()
    h1 = seguimiento.spawn(data, inicio, fin, etiqueta)
    h2 = calibrar.spawn(data, inicio, fin, etiqueta)
    s = h1.get()
    c = h2.get()
    json.dump(s["tracks"], open(os.path.join(salida, "tracks.json"), "w"))
    json.dump(s["balls"], open(os.path.join(salida, "balls.json"), "w"))
    json.dump(s["stats"], open(os.path.join(salida, "stats_seguimiento.json"), "w"), indent=1)
    json.dump(c, open(os.path.join(salida, "calibracion.json"), "w"))
    ok = sum(p is not None for p in c["params"])
    print(json.dumps({"seguimiento": s["stats"], "calibradas": ok, "imagenes": len(c["params"]),
                      "seg_calib_promedio": sum(c["seg_por_imagen"]) / len(c["seg_por_imagen"])}))
