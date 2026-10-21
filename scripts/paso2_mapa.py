"""Paso 2b: posiciones en metros y mapa de la cancha visto desde arriba.

Usa la calibración de PnLCalib y el seguimiento (Deep-EIoU) de modal_apps/paso2_completo.py.
La posición de cada jugador es el punto medio del borde inferior de su recuadro (los pies),
llevado al plano de la cancha (altura 0) con la matriz de proyección de cada imagen.

Control de calidad de la calibración: la cámara principal de TV está fija en el estadio,
así que su posición estimada debería ser casi constante. Se descartan las imágenes cuya
posición de cámara se aleja más de UMBRAL_CAMARA_M metros de la mediana del tramo.

Uso:
  python scripts/paso2_mapa.py VIDEO RESULTADOS_DIR NUEVE SALIDA_DIR
  NUEVE: identidades del 9 por tramo de imágenes, p. ej. "24@83-238".

Coordenadas: metros, origen en el centro de la cancha, cancha de 105 × 68 m
(convención de PnLCalib). X a lo largo, Y a lo ancho.
"""
import json
import os
import sys

import cv2
import numpy as np

LARGO, ANCHO = 105.0, 68.0
UMBRAL_CAMARA_M = 5.0
FPS = 30000 / 1001


def parse_nueve(s):
    tramos = []
    for p in s.split(","):
        tid, rng = p.split("@")
        a, b = rng.split("-")
        tramos.append((tid, int(a), int(b)))
    return tramos


def id_nueve(tramos, f):
    for tid, a, b in tramos:
        if a <= f <= b:
            return tid
    return None


def a_cancha(P, u, v):
    H = np.asarray(P)[:, [0, 1, 3]]
    w = np.linalg.solve(H, np.array([u, v, 1.0]))
    return float(w[0] / w[2]), float(w[1] / w[2])


def lineas_cancha():
    """Segmentos de las líneas de la cancha (metros, altura 0)."""
    L, A = LARGO / 2, ANCHO / 2
    seg = [((-L, -A), (L, -A)), ((-L, A), (L, A)), ((-L, -A), (-L, A)), ((L, -A), (L, A)),
           ((0, -A), (0, A))]
    for s in (-1, 1):
        x0, xa, xb = s * L, s * (L - 16.5), s * (L - 5.5)
        seg += [((x0, -20.16), (xa, -20.16)), ((x0, 20.16), (xa, 20.16)), ((xa, -20.16), (xa, 20.16)),
                ((x0, -9.16), (xb, -9.16)), ((x0, 9.16), (xb, 9.16)), ((xb, -9.16), (xb, 9.16))]
    circ = [(9.15 * np.cos(t), 9.15 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 73)]
    seg += list(zip(circ[:-1], circ[1:]))
    return seg


def dibujar_lineas(img, P):
    P = np.asarray(P)
    for (x1, y1), (x2, y2) in lineas_cancha():
        a = P @ np.array([x1, y1, 0.0, 1.0])
        b = P @ np.array([x2, y2, 0.0, 1.0])
        if a[2] <= 0 or b[2] <= 0:
            continue
        pa = (int(a[0] / a[2]), int(a[1] / a[2]))
        pb = (int(b[0] / b[2]), int(b[1] / b[2]))
        cv2.line(img, pa, pb, (255, 0, 0), 2)
    return img


def dibujar_mapa(escala=6, margen=30):
    W, H = int(LARGO * escala) + 2 * margen, int(ANCHO * escala) + 2 * margen
    img = np.full((H, W, 3), (40, 110, 40), np.uint8)

    def px(x, y):
        return int(round((x + LARGO / 2) * escala + margen)), int(round((y + ANCHO / 2) * escala + margen))

    for a, b in lineas_cancha():
        cv2.line(img, px(*a), px(*b), (230, 230, 230), 2)
    return img, px


def main(video, resdir, nueve, outdir):
    os.makedirs(outdir, exist_ok=True)
    cal = json.load(open(os.path.join(resdir, "calibracion.json")))
    tracks = json.load(open(os.path.join(resdir, "tracks.json")))
    inicio = cal["inicio"]
    tramos = parse_nueve(nueve)
    n = len(cal["params"])

    # Control de calidad de la calibración
    pos_cam = np.array([p["cam_params"]["position_meters"] if p else [np.nan] * 3 for p in cal["params"]])
    mediana = np.nanmedian(pos_cam, axis=0)
    dist_cam = np.linalg.norm(pos_cam - mediana, axis=1)
    valida = [cal["params"][i] is not None and dist_cam[i] <= UMBRAL_CAMARA_M for i in range(n)]

    posiciones = {}
    for i in range(n):
        if not valida[i]:
            continue
        f = inicio + i
        fila = {}
        for tid, tr in tracks.items():
            b = tr.get(str(f))
            if b is None:
                continue
            x1, y1, x2, y2, _, interp = b
            fila[tid] = [*a_cancha(cal["params"][i]["P"], (x1 + x2) / 2, y2), bool(interp)]
        posiciones[f] = fila
    json.dump({"umbral_camara_m": UMBRAL_CAMARA_M, "imagenes_validas": [inicio + i for i in range(n) if valida[i]],
               "posiciones": posiciones}, open(os.path.join(outdir, "posiciones_m.json"), "w"))

    # Video: imagen con líneas proyectadas a la izquierda, mapa a la derecha
    cap = cv2.VideoCapture(video)
    for _ in range(inicio):
        cap.read()
    base, px = dibujar_mapa()
    alto = base.shape[0]
    out = None
    estela = []
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        f = inicio + i
        if valida[i]:
            frame = dibujar_lineas(frame, cal["params"][i]["P"])
        izq = cv2.resize(frame, (int(frame.shape[1] * alto / frame.shape[0]), alto))
        mapa = base.copy()
        nid = id_nueve(tramos, f)
        if not valida[i]:
            cv2.putText(mapa, "calibracion descartada", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            for tid, (x, y, _) in posiciones[f].items():
                if tid == nid:
                    continue
                cv2.circle(mapa, px(x, y), 6, (255, 255, 255), -1)
                cv2.putText(mapa, tid, (px(x, y)[0] + 7, px(x, y)[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if nid in posiciones[f]:
                x, y, _ = posiciones[f][nid]
                estela.append(px(x, y))
                cv2.circle(mapa, px(x, y), 9, (0, 0, 255), -1)
                cv2.putText(mapa, "9", (px(x, y)[0] + 10, px(x, y)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for a, b in zip(estela[:-1], estela[1:]):
            cv2.line(mapa, a, b, (0, 0, 255), 1)
        cv2.putText(mapa, f"imagen {f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        lienzo = np.hstack([izq, mapa])
        if out is None:
            out = cv2.VideoWriter(os.path.join(outdir, "paso2_mapa.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                  FPS, (lienzo.shape[1], lienzo.shape[0]))
        out.write(lienzo)
    out.release()

    # Controles de calidad numéricos
    todos = [(x, y) for fila in posiciones.values() for x, y, _ in fila.values()]
    dentro = sum(abs(x) <= LARGO / 2 + 2 and abs(y) <= ANCHO / 2 + 2 for x, y in todos)
    vel = []
    for tid, tr in tracks.items():
        frames = sorted(int(k) for k in tr)
        for a, b in zip(frames[:-1], frames[1:]):
            if b == a + 1 and a in posiciones and b in posiciones and tid in posiciones[a] and tid in posiciones[b]:
                pa, pb = posiciones[a][tid], posiciones[b][tid]
                vel.append(float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]) * FPS))
    vel = np.array(vel)
    stats = {
        "imagenes": n,
        "calibradas_por_pnlcalib": int(sum(p is not None for p in cal["params"])),
        "calibraciones_validas": int(sum(valida)),
        "camara_mediana_m": mediana.round(2).tolist(),
        "posiciones_jugador": len(todos),
        "posiciones_dentro_cancha_margen_2m": int(dentro),
        "velocidad_imagen_a_imagen_mps": {
            "n": int(len(vel)), "mediana": float(np.median(vel)), "p90": float(np.percentile(vel, 90)),
            "p99": float(np.percentile(vel, 99)), "max": float(vel.max()),
            "sobre_12_mps": int((vel > 12).sum())},
        "nueve_imagenes_con_posicion": int(sum(1 for f, fila in posiciones.items() if id_nueve(tramos, f) in fila)),
    }
    json.dump(stats, open(os.path.join(outdir, "stats_mapa.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
