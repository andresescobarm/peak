"""Benchmark CPU de los modelos de la Fase 1.
Mide tiempo de carga y segundos por cuadro sobre cuadros reales de video con personas
(video de ejemplo incluido en el repositorio oficial de SAM 2: notebooks/videos/bedroom).
"""
import glob, json, time, traceback, warnings, os
warnings.filterwarnings("ignore")
import numpy as np, cv2, torch

torch.set_num_threads(os.cpu_count())
FRAMES = sorted(glob.glob("/home/claude/sam2/notebooks/videos/bedroom/*.jpg"))[:20]
imgs_bgr = [cv2.imread(f) for f in FRAMES]
imgs_rgb = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs_bgr]
H, W = imgs_bgr[0].shape[:2]
N_MEASURE = 10
results = []

def add(func, tool, model, status, load_s=None, per_frame_s=None, note=""):
    r = dict(funcion=func, herramienta=tool, modelo=model, estado=status,
             carga_s=None if load_s is None else round(load_s, 2),
             seg_por_cuadro=None if per_frame_s is None else round(per_frame_s, 3),
             nota=note)
    results.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)

# ---------- 1. Detección: RF-DETR (Apache 2.0: N, S, M) ----------
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
det_medium = None
for cls, name in [(RFDETRNano, "RF-DETR Nano"), (RFDETRSmall, "RF-DETR Small"), (RFDETRMedium, "RF-DETR Medium")]:
    try:
        t0 = time.perf_counter(); m = cls(device="cpu"); load = time.perf_counter() - t0
        m.predict(imgs_rgb[0], threshold=0.5)  # calentamiento
        times, persons = [], []
        for im in imgs_rgb[1:1 + N_MEASURE]:
            t0 = time.perf_counter(); d = m.predict(im, threshold=0.5); times.append(time.perf_counter() - t0)
            persons.append(int((d.class_id == 1).sum()) if d.class_id is not None else 0)
        add("Detección jugadores/balón", "RF-DETR 1.11.2", name, "CORRE", load, float(np.mean(times)),
            f"personas detectadas por cuadro (min-max): {min(persons)}-{max(persons)}")
        if name == "RF-DETR Medium": det_medium = m
    except Exception as e:
        add("Detección jugadores/balón", "RF-DETR 1.11.2", name, "FALLA", note=repr(e)[:300]); traceback.print_exc()

# Detecciones de personas para el resto del pipeline
dets = []
if det_medium is not None:
    for im in imgs_rgb:
        d = det_medium.predict(im, threshold=0.5)
        dets.append(d[d.class_id == 1])

# ---------- 2. Tracking: trackers ByteTrack ----------
try:
    from trackers import ByteTrackTracker
    tr = ByteTrackTracker(frame_rate=24.0)
    times, ids = [], set()
    for d in dets:
        t0 = time.perf_counter(); out = tr.update(d); times.append(time.perf_counter() - t0)
        if out.tracker_id is not None: ids.update(int(x) for x in out.tracker_id if x is not None and x >= 0)
    add("Seguimiento de todos los jugadores", "trackers 2.6.1", "ByteTrack", "CORRE", 0.0, float(np.mean(times)),
        f"IDs de seguimiento asignados en {len(dets)} cuadros: {len(ids)}")
except Exception as e:
    add("Seguimiento de todos los jugadores", "trackers 2.6.1", "ByteTrack", "FALLA", note=repr(e)[:300]); traceback.print_exc()

# ---------- 3. Pose: rtmlib RTMPose (cajas dadas por RF-DETR) ----------
POSE = {
    "RTMPose-s body7": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip", (192, 256)),
    "RTMPose-m body7": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip", (192, 256)),
    "RTMPose-x body7": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip", (288, 384)),
}
from rtmlib import RTMPose
for name, (url, size) in POSE.items():
    try:
        t0 = time.perf_counter(); pm = RTMPose(onnx_model=url, model_input_size=size, backend="onnxruntime", device="cpu"); load = time.perf_counter() - t0
        box = dets[0].xyxy[:1].tolist()
        pm(imgs_bgr[0], bboxes=box)  # calentamiento
        times = []; kp_shape = None
        for i in range(1, 1 + N_MEASURE):
            b = dets[i].xyxy[:1].tolist()
            t0 = time.perf_counter(); kps, sc = pm(imgs_bgr[i], bboxes=b); times.append(time.perf_counter() - t0)
            kp_shape = kps.shape
        add("Pose del target (1 jugador)", "rtmlib 0.0.16", name, "CORRE", load, float(np.mean(times)),
            f"puntos corporales por jugador: {kp_shape[1]}")
    except Exception as e:
        add("Pose del target (1 jugador)", "rtmlib 0.0.16", name, "FALLA", note=repr(e)[:300]); traceback.print_exc()

# ---------- 4. Seguimiento del target por clic: SAM 2.1 (transformers) ----------
from transformers import Sam2VideoModel, Sam2VideoProcessor
for hf_id, name in [("facebook/sam2.1-hiera-tiny", "SAM 2.1 tiny"), ("facebook/sam2.1-hiera-small", "SAM 2.1 small")]:
    try:
        t0 = time.perf_counter()
        model = Sam2VideoModel.from_pretrained(hf_id).to("cpu").eval()
        proc = Sam2VideoProcessor.from_pretrained(hf_id)
        load = time.perf_counter() - t0
        x1, y1, x2, y2 = dets[0].xyxy[0].tolist()
        click = [float((x1 + x2) / 2), float((y1 + y2) / 2)]
        sess = proc.init_video_session(video=imgs_rgb, inference_device="cpu", dtype=torch.float32)
        proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=0, obj_ids=1,
                                             input_points=[[[click]]], input_labels=[[[1]]])
        times = []; areas = []
        with torch.inference_mode():
            model(inference_session=sess, frame_idx=0)
            t_prev = time.perf_counter()
            for out in model.propagate_in_video_iterator(sess):
                now = time.perf_counter(); times.append(now - t_prev); t_prev = now
                m = proc.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                areas.append(int(m.sum()))
        per = float(np.mean(times[1:])) if len(times) > 1 else float(np.mean(times))
        nz = sum(1 for a in areas if a > 0)
        add("Seguir al target desde un clic", "transformers 5.20.2", name, "CORRE", load, per,
            f"cuadros con el target segmentado: {nz} de {len(areas)}")
    except Exception as e:
        add("Seguir al target desde un clic", "transformers 5.20.2", name, "FALLA", note=repr(e)[:300]); traceback.print_exc()

json.dump(dict(frames=len(FRAMES), size=[W, H], cpu_threads=os.cpu_count(), results=results),
          open("/home/claude/bench/results.json", "w"), ensure_ascii=False, indent=1)
print("FIN")
