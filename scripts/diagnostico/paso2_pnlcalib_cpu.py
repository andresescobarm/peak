"""Diagnóstico local (CPU) de PnLCalib: integración y variantes de pesos."""
import os, sys, time, json
import numpy as np, cv2, torch, yaml
os.chdir("/tmp/pnl_src/PnLCalib"); sys.path.insert(0, "/tmp/pnl_src/PnLCalib")
import inference as inf
from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
import torchvision.transforms as T
torch.set_num_threads(2)
inf.device = "cpu"; inf.transform2 = T.Resize((540, 960))
W = "/home/claude/pnl_weights"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

def load(variant):
    cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml")); cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml"))
    m = get_cls_net(cfg); m.load_state_dict(torch.load(f"{W}/{variant}_kp", map_location="cpu")); m.eval()
    ml = get_cls_net_l(cfg_l); ml.load_state_dict(torch.load(f"{W}/{variant}_lines", map_location="cpu")); ml.eval()
    return m, ml

def run(frames, m, ml, refine, tag, kp_t=0.3434, ln_t=0.7867):
    h, w = frames[0].shape[:2]
    res = []
    for k, fr in enumerate(frames):
        cam = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
        p = inf.inference(cam, fr.copy(), m, ml, kp_t, ln_t, refine)
        ok = p is not None
        img = inf.project(fr.copy(), inf.projection_from_cam_params(p)) if ok else fr.copy()
        cv2.imwrite(f"{OUT}/{tag}_{k}.jpg", img)
        res.append(ok)
    return res

mode = sys.argv[2]
if mode == "ejemplo":
    m, ml = load("SV")
    pass
    msg = cv2.imread("examples/messi_sample.png")
    print("messi", msg.shape, run([msg], m, ml, True, "messi"))
else:
    raw = np.load("/home/claude/clip/jugada_530_866.npz")["frames"]
    idx = [int(x) for x in sys.argv[3].split(",")]
    frs = [raw[i] for i in idx]
    for variant in sys.argv[4].split(","):
        m, ml = load(variant)
        for refine in (True, False):
            t = time.time()
            r = run(frs, m, ml, refine, f"{variant}_{'pnl' if refine else 'base'}")
            print(variant, "refine" if refine else "base", dict(zip(idx, r)), f"{time.time()-t:.0f}s", flush=True)
