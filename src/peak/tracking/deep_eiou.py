"""Implementación propia del método Deep-EIoU.

Escrita desde el paper, sin usar el código de los autores:
Huang et al., "Iterative Scale-Up ExpansionIoU and Deep Features Association
for Multi-Object Tracking in Sports", WACV 2024 Workshops (arXiv 2306.13074).

Parámetros tomados del paper:
- Sin filtro de Kalman: se asocia contra la última caja de cada trayectoria.
- Expansion IoU: ancho y alto de cada caja se multiplican por (2E + 1), con el mismo centro.
- Escala iterativa en la asociación de detecciones de alta confianza: E_t = 0,7 + 0,1·t.
- Costo de apariencia C_A = 1 - coseno; compuerta: C_Â = 1 si C_A > 0,25 o C_EIoU > 0,5; si no, 0,5·C_A.
- Costo etapa 1 = min(C_Â, C_EIoU). Etapa 2 (confianza baja, 0,1 a 0,6) = C_EIoU con E = 0,7.
- Detecciones no asociadas: E = 0,5. Trayectorias perdidas se conservan 60 imágenes.
- Interpolación lineal de huecos después del seguimiento.

Parámetros que el paper no especifica (valores por defecto de ByteTrack/BoT-SORT, marcados como SUPUESTO):
- Número de iteraciones de la escala: 2 (t = 0 y t = 1).
- Umbral de aceptación del emparejamiento: 0,8 en etapa 1, 0,5 en etapa 2, 0,7 para trayectorias sin confirmar.
- Umbral para crear una trayectoria nueva: 0,7.
- Actualización de rasgos de apariencia: promedio móvil con alfa 0,9.

Rasgos de apariencia: solo se usan en la etapa 1 (confianza alta). Una fila de NaN en `feats`
indica que no se calculó para esa detección (por ejemplo, detecciones de confianza baja) y no
se usa para actualizar la trayectoria.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

HIGH_THRESH = 0.6          # paper
LOW_THRESH = 0.1           # paper
E_INITIAL = 0.7            # paper
E_STEP = 0.1               # paper
E_ITERS = 2                # SUPUESTO (paper: t_total = 2)
E_UNMATCHED = 0.5          # paper
TAU_APPEARANCE = 0.25      # paper
TAU_EIOU = 0.5             # paper
TRACK_BUFFER = 60          # paper
MATCH_THRESH_1 = 0.8       # SUPUESTO (ByteTrack)
MATCH_THRESH_2 = 0.5       # SUPUESTO (ByteTrack)
MATCH_THRESH_UNCONF = 0.7  # SUPUESTO (ByteTrack)
NEW_TRACK_THRESH = 0.7     # SUPUESTO (ByteTrack: umbral alto + 0,1)
FEAT_ALPHA = 0.9           # SUPUESTO (BoT-SORT)


def expand(boxes: np.ndarray, e: float) -> np.ndarray:
    """Expande cajas xyxy: ancho y alto × (2e + 1), mismo centro."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = (boxes[:, 2] - boxes[:, 0]) * (2 * e + 1)
    h = (boxes[:, 3] - boxes[:, 1]) * (2 * e + 1)
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def eiou_cost(track_boxes: np.ndarray, det_boxes: np.ndarray, e: float) -> np.ndarray:
    return 1.0 - iou_matrix(expand(track_boxes, e), expand(det_boxes, e))


def appearance_cost(track_feats: np.ndarray, det_feats: np.ndarray) -> np.ndarray:
    if len(track_feats) == 0 or len(det_feats) == 0:
        return np.ones((len(track_feats), len(det_feats)))
    a = track_feats / (np.linalg.norm(track_feats, axis=1, keepdims=True) + 1e-9)
    b = det_feats / (np.linalg.norm(det_feats, axis=1, keepdims=True) + 1e-9)
    return np.clip(1.0 - a @ b.T, 0.0, 2.0)


def hungarian(cost: np.ndarray, thresh: float):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    mr = {r for r, _ in matches}
    mc = {c for _, c in matches}
    return (matches,
            [r for r in range(cost.shape[0]) if r not in mr],
            [c for c in range(cost.shape[1]) if c not in mc])


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    score: float
    feat: np.ndarray | None
    frame: int
    confirmed: bool
    lost: bool = False
    history: list = field(default_factory=list)  # (frame, box, score)

    def update(self, box, score, feat, frame):
        self.box, self.score, self.frame = np.asarray(box, float), float(score), frame
        if feat is not None and np.all(np.isfinite(feat)):
            f = feat / (np.linalg.norm(feat) + 1e-9)
            self.feat = f if self.feat is None else FEAT_ALPHA * self.feat + (1 - FEAT_ALPHA) * f
            self.feat = self.feat / (np.linalg.norm(self.feat) + 1e-9)
        self.lost = False
        self.history.append((frame, self.box.copy(), self.score))


class DeepEIoUTracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self.removed: list[Track] = []
        self._next_id = 1
        self._first_frame = True

    def _new_track(self, box, score, feat, frame, confirmed):
        f = None if feat is None else feat / (np.linalg.norm(feat) + 1e-9)
        t = Track(self._next_id, np.asarray(box, float), float(score), f, frame, confirmed)
        t.history.append((frame, t.box.copy(), t.score))
        self._next_id += 1
        self.tracks.append(t)

    def update(self, boxes: np.ndarray, scores: np.ndarray, feats: np.ndarray, frame: int):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        feats = np.asarray(feats, float).reshape(len(boxes), -1) if len(boxes) else np.zeros((0, 1))
        keep = scores > LOW_THRESH
        boxes, scores, feats = boxes[keep], scores[keep], feats[keep]
        hi = np.where(scores > HIGH_THRESH)[0]
        lo = np.where(scores <= HIGH_THRESH)[0]

        confirmed = [t for t in self.tracks if t.confirmed]
        unconfirmed = [t for t in self.tracks if not t.confirmed]

        # Etapa 1: confianza alta, escala iterativa, apariencia + EIoU
        pool = list(confirmed)
        rem_hi = list(hi)
        for it in range(E_ITERS):
            if not pool or not rem_hi:
                break
            e = E_INITIAL + E_STEP * it
            tb = np.stack([t.box for t in pool])
            c_eiou = eiou_cost(tb, boxes[rem_hi], e)
            c_a = appearance_cost(np.stack([t.feat for t in pool]), feats[rem_hi])
            c_hat = np.where((c_a > TAU_APPEARANCE) | (c_eiou > TAU_EIOU), 1.0, 0.5 * c_a)
            cost = np.minimum(c_hat, c_eiou)
            matches, um_t, um_d = hungarian(cost, MATCH_THRESH_1)
            for r, c in matches:
                d = rem_hi[c]
                pool[r].update(boxes[d], scores[d], feats[d], frame)
            pool = [pool[r] for r in um_t]
            rem_hi = [rem_hi[c] for c in um_d]

        # Etapa 2: confianza baja, solo EIoU, solo trayectorias activas (no perdidas)
        active = [t for t in pool if not t.lost]
        if active and len(lo):
            cost = eiou_cost(np.stack([t.box for t in active]), boxes[lo], E_INITIAL)
            matches, um_t, _ = hungarian(cost, MATCH_THRESH_2)
            matched = {id(active[r]) for r, _ in matches}
            for r, c in matches:
                d = lo[c]
                active[r].update(boxes[d], scores[d], feats[d], frame)
            pool = [t for t in pool if id(t) not in matched]

        for t in pool:
            t.lost = True

        # Trayectorias sin confirmar contra detecciones altas restantes
        if unconfirmed and rem_hi:
            cost = eiou_cost(np.stack([t.box for t in unconfirmed]), boxes[rem_hi], E_UNMATCHED)
            matches, um_t, um_d = hungarian(cost, MATCH_THRESH_UNCONF)
            for r, c in matches:
                d = rem_hi[c]
                unconfirmed[r].update(boxes[d], scores[d], feats[d], frame)
                unconfirmed[r].confirmed = True
            dead = {id(unconfirmed[r]) for r in um_t}
            self.tracks = [t for t in self.tracks if id(t) not in dead]
            rem_hi = [rem_hi[c] for c in um_d]
        elif unconfirmed:
            dead = {id(t) for t in unconfirmed}
            self.tracks = [t for t in self.tracks if id(t) not in dead]

        # Trayectorias nuevas
        for d in rem_hi:
            if scores[d] >= NEW_TRACK_THRESH:
                self._new_track(boxes[d], scores[d], feats[d], frame, confirmed=self._first_frame)
        self._first_frame = False

        # Eliminar perdidas más allá del buffer
        alive = []
        for t in self.tracks:
            if t.lost and frame - t.frame > TRACK_BUFFER:
                self.removed.append(t)
            else:
                alive.append(t)
        self.tracks = alive

        return [t for t in self.tracks if t.confirmed and not t.lost and t.frame == frame]

    def all_tracks(self) -> list[Track]:
        return [t for t in self.tracks + self.removed if t.confirmed]


def interpolate(history: list) -> dict:
    """Interpolación lineal de huecos: {frame: (box, score, interpolada)}."""
    out = {}
    hist = sorted(history, key=lambda h: h[0])
    for (f0, b0, s0), (f1, b1, s1) in zip(hist[:-1], hist[1:]):
        out[f0] = (b0, s0, False)
        for f in range(f0 + 1, f1):
            a = (f - f0) / (f1 - f0)
            out[f] = ((1 - a) * b0 + a * b1, min(s0, s1), True)
    if hist:
        f, b, s = hist[-1]
        out[f] = (b, s, False)
    return out
