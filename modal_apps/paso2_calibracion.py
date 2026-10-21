"""Paso 2: calibración de cancha con PnLCalib, ejecutada en Modal (GPU T4).

Recibe las imágenes de la jugada (npz con arreglo `frames`, BGR, N×H×W×3) y devuelve,
por imagen, los parámetros de cámara estimados por PnLCalib y la imagen con las líneas
de la cancha proyectadas.

Uso (desde la raíz del repositorio):
  modal run modal_apps/paso2_calibracion.py --entrada JUGADA.npz --salida resultados/paso2_clip1997

Límites de costo: timeout de 900 s, GPU T4, 2 núcleos, 16 GiB.
"""
import io
import json
import os

import modal

PNLCALIB_COMMIT = "8c87391d6f4ea40c5e4d65e61529916c7a49ce62"
PESOS_URL = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "scipy==1.13.1",
        "shapely==2.0.7",
        "lsq-ellipse==2.2.1",
        "PyYAML==6.0.2",
        "matplotlib==3.9.4",
        "tqdm==4.67.1",
    )
    .run_commands(
        "git clone https://github.com/mguti97/PnLCalib /opt/PnLCalib",
        f"cd /opt/PnLCalib && git checkout {PNLCALIB_COMMIT}",
        "mkdir -p /opt/PnLCalib/weights",
        f"wget -q -O /opt/PnLCalib/weights/SV_kp {PESOS_URL}/SV_kp",
        f"wget -q -O /opt/PnLCalib/weights/SV_lines {PESOS_URL}/SV_lines",
    )
)

app = modal.App("peak-paso2-calibracion", image=image)
# Copia de respaldo de los resultados en Modal, por si falla la descarga.
vol = modal.Volume.from_name("peak-resultados", create_if_missing=True)


@app.function(gpu="T4", cpu=2.0, memory=16384, timeout=900, volumes={"/resultados": vol})
def calibrar(npz_bytes: bytes, pnl_refine: bool = True) -> dict:
    import sys
    import time

    import numpy as np
    import torch
    import yaml

    os.chdir("/opt/PnLCalib")
    sys.path.insert(0, "/opt/PnLCalib")
    import inference as inf  # funciones oficiales de PnLCalib
    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l
    from utils.utils_calib import FramebyFrameCalib
    import torchvision.transforms as T

    frames = np.load(io.BytesIO(npz_bytes))["frames"]
    n, h, w, _ = frames.shape
    device = "cuda:0"

    t0 = time.perf_counter()
    cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml"))
    cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml"))
    model = get_cls_net(cfg)
    model.load_state_dict(torch.load("weights/SV_kp", map_location=device))
    model.to(device).eval()
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load("weights/SV_lines", map_location=device))
    model_l.to(device).eval()
    # inference.inference() usa estas variables globales del módulo
    inf.device = device
    inf.transform2 = T.Resize((540, 960))
    t_carga = time.perf_counter() - t0

    cam = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
    params, proyectadas, tiempos = [], [], []
    for i in range(n):
        t1 = time.perf_counter()
        p = inf.inference(cam, frames[i].copy(), model, model_l,
                          0.3434, 0.7867, pnl_refine)
        tiempos.append(time.perf_counter() - t1)
        if p is not None:
            P = inf.projection_from_cam_params(p)
            proyectadas.append(inf.project(frames[i].copy(), P))
            params.append({"cam_params": p["cam_params"], "P": P.tolist()})
        else:
            proyectadas.append(frames[i].copy())
            params.append(None)

    buf = io.BytesIO()
    np.savez_compressed(buf, frames=np.stack(proyectadas))
    out = {
        "params": params,
        "seg_carga_modelos": t_carga,
        "seg_por_imagen": tiempos,
        "gpu": torch.cuda.get_device_name(0),
    }
    os.makedirs("/resultados/paso2", exist_ok=True)
    with open("/resultados/paso2/calibracion.json", "w") as fh:
        json.dump(out, fh)
    with open("/resultados/paso2/proyectadas.npz", "wb") as fh:
        fh.write(buf.getvalue())
    vol.commit()
    out["proyectadas_npz"] = buf.getvalue()
    return out


@app.local_entrypoint()
def main(entrada: str, salida: str):
    os.makedirs(salida, exist_ok=True)
    with open(entrada, "rb") as fh:
        r = calibrar.remote(fh.read())
    with open(os.path.join(salida, "proyectadas.npz"), "wb") as fh:
        fh.write(r.pop("proyectadas_npz"))
    with open(os.path.join(salida, "calibracion.json"), "w") as fh:
        json.dump(r, fh)
    ok = sum(p is not None for p in r["params"])
    t = r["seg_por_imagen"]
    print(json.dumps({"gpu": r["gpu"], "imagenes": len(t), "calibradas": ok,
                      "seg_carga_modelos": round(r["seg_carga_modelos"], 2),
                      "seg_por_imagen_promedio": round(sum(t) / len(t), 4)}))
