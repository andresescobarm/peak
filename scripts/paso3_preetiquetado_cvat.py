"""Paso 3: pre-etiquetado para CVAT (formato "CVAT for video 1.1") desde el seguimiento.

Convierte tracks.json (Deep-EIoU) y balls.json (RF-DETR) en trayectorias de CVAT.
Andrés corrige en CVAT; el resultado corregido es la referencia para los pasos 4 y 6.

Uso:
  python scripts/paso3_preetiquetado_cvat.py RESULTADOS_DIR ROLES SALIDA_XML ULTIMA_IMAGEN ANCHO ALTO
  ROLES: etiqueta por identidad distinta de "jugador", p. ej. "1=arbitro,18=arbitro,25=arquero".

Reglas:
- Una trayectoria por identidad. Imágenes vistas por el detector: keyframe=1.
  Imágenes interpoladas por el seguimiento: keyframe=0 (CVAT las recalcula).
- Fin de trayectoria: recuadro con outside=1 en la imagen siguiente a la última.
- Balón: detección de mayor confianza por imagen, con confianza >= 0,3; una trayectoria
  por cada tramo de imágenes consecutivas.
"""
import json
import os
import sys
from xml.sax.saxutils import quoteattr

UMBRAL_BALON = 0.3
ETIQUETAS = ["jugador", "arquero", "arbitro", "balon"]


def caja(frame, b, outside=0, keyframe=1):
    x1, y1, x2, y2 = b
    return (f'    <box frame="{frame}" outside="{outside}" occluded="0" keyframe="{keyframe}" '
            f'xtl="{x1:.2f}" ytl="{y1:.2f}" xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0"></box>')


def main(resdir, roles_spec, salida, ultima_imagen, ancho, alto):
    tracks = json.load(open(os.path.join(resdir, "tracks.json")))
    balls = json.load(open(os.path.join(resdir, "balls.json")))
    roles = dict(p.split("=") for p in roles_spec.split(",")) if roles_spec else {}
    lineas = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>",
              "  <meta>", "    <task>", f"      <size>{ultima_imagen + 1}</size>", "      <mode>interpolation</mode>",
              f"      <original_size><width>{ancho}</width><height>{alto}</height></original_size>",
              "      <labels>"]
    for nombre in ETIQUETAS:
        lineas.append(f"        <label><name>{nombre}</name><type>rectangle</type><attributes></attributes></label>")
    lineas += ["      </labels>", "    </task>", "  </meta>"]
    nid = 0
    resumen = {"jugador": 0, "arquero": 0, "arbitro": 0, "balon": 0}
    for tid in sorted(tracks, key=int):
        tr = tracks[tid]
        etiqueta = roles.get(tid, "jugador")
        frames = sorted(int(f) for f in tr)
        lineas.append(f'  <track id="{nid}" label={quoteattr(etiqueta)} source="auto">')
        for f in frames:
            v = tr[str(f)]
            lineas.append(caja(f, v[:4], keyframe=0 if v[5] else 1))
        if frames[-1] < ultima_imagen:
            lineas.append(caja(frames[-1] + 1, tr[str(frames[-1])][:4], outside=1))
        lineas.append("  </track>")
        nid += 1
        resumen[etiqueta] += 1
    # Balón
    mejor = {}
    for f, dets in balls.items():
        dets = [d for d in dets if d[4] >= UMBRAL_BALON]
        if dets:
            mejor[int(f)] = max(dets, key=lambda d: d[4])[:4]
    tramos, actual = [], []
    for f in sorted(mejor):
        if actual and f != actual[-1] + 1:
            tramos.append(actual)
            actual = []
        actual.append(f)
    if actual:
        tramos.append(actual)
    for t in tramos:
        lineas.append(f'  <track id="{nid}" label="balon" source="auto">')
        for f in t:
            lineas.append(caja(f, mejor[f]))
        if t[-1] < ultima_imagen:
            lineas.append(caja(t[-1] + 1, mejor[t[-1]], outside=1))
        lineas.append("  </track>")
        nid += 1
        resumen["balon"] += 1
    lineas.append("</annotations>")
    open(salida, "w").write("\n".join(lineas) + "\n")
    print(json.dumps({"trayectorias": resumen, "imagenes_con_balon": len(mejor)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
