import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from peak.tracking.deep_eiou import DeepEIoUTracker, expand, interpolate


def test_expand_keeps_center_and_scales():
    b = np.array([[10, 20, 30, 60]], float)  # w=20, h=40
    e = expand(b, 0.7)[0]                     # factor 2*0.7+1 = 2.4
    assert np.allclose([(e[0] + e[2]) / 2, (e[1] + e[3]) / 2], [20, 40])
    assert np.isclose(e[2] - e[0], 48) and np.isclose(e[3] - e[1], 96)


def test_two_crossing_players_keep_ids():
    rng = np.random.default_rng(0)
    fa, fb = rng.normal(size=512), rng.normal(size=512)
    tr = DeepEIoUTracker()
    ids = {}
    for f in range(40):
        a = [100 + 4 * f, 100, 120 + 4 * f, 150]   # se mueve a la derecha
        b = [260 - 4 * f, 102, 280 - 4 * f, 152]   # se mueve a la izquierda, se cruzan
        out = tr.update(np.array([a, b]), np.array([0.9, 0.9]),
                        np.stack([fa + rng.normal(scale=0.05, size=512),
                                  fb + rng.normal(scale=0.05, size=512)]), f)
        for t in out:
            x = (t.box[0] + t.box[2]) / 2
            ids.setdefault(t.track_id, []).append(x)
    assert len(ids) == 2
    # cada ID debe moverse siempre en la misma dirección (no se intercambiaron)
    dirs = sorted(np.sign(np.diff(v)).mean() for v in ids.values())
    assert dirs[0] < -0.9 and dirs[1] > 0.9


def test_occlusion_recovered_and_interpolated():
    feat = np.ones(512)
    tr = DeepEIoUTracker()
    for f in range(30):
        if 10 <= f < 15:
            tr.update(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 512)), f)  # tapado
            continue
        tr.update(np.array([[100 + f, 100, 120 + f, 150]]), np.array([0.9]), feat[None], f)
    tracks = tr.all_tracks()
    assert len(tracks) == 1
    full = interpolate(tracks[0].history)
    assert all(f in full for f in range(30))
    assert all(full[f][2] for f in range(10, 15))
