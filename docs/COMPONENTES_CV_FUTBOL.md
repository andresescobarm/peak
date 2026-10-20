# Visión computacional para fútbol — componentes óptimos, alternativas, licencias y costos

Investigación: 20-oct-2026. Alcance: solo análisis de video (detección, seguimiento, posiciones en cancha, movimientos del cuerpo, acciones). Cada dato lleva su fuente al final del documento.

> "Si no te puedo descomponer este número en partes que puedas revisar una por una, no es un número: es una impresión, y debes tratarla como tal."
> Todos los números de este documento son cifras publicadas por la fuente indicada en la misma fila o párrafo. No hay estimaciones propias, salvo las conversiones de precio por segundo a precio por hora, marcadas como tales.

**Criterios, en orden:** 1) precisión en fútbol, usando los benchmarks de fútbol (SoccerNet, SoccerTrack) cuando existen; 2) licencia verificable para uso comercial, gratis o pagada; 3) costo y operación.

---

## 1. Referencia: cómo resuelven esto los mejores sistemas de fútbol hoy

| Benchmark | Ganador | Puntaje | Componentes |
|---|---|---|---|
| SoccerNet Game State Reconstruction 2025 (posición e identidad de cada jugador en el plano de la cancha, desde transmisión de TV) | KIST-GSR | GS-HOTA 63,90 (2º: Constructor.Tech, 63,81) | Detector YOLO-X · Tracker Deep-EIoU + división/unión de trayectorias · Re-ID OSNet · Calibración con modelo de puntos clave multi-cuadro · Camiseta, rol y color con LLaMA 3.2 Vision · Equipo por clustering de color [S1] |
| SoccerTrack 2025 (seguimiento con cámaras fijas ojo de pez, 4096×1080) | GTATrack | HOTA 0,60 vs 0,42 de ByteTrack (tabla 3 del paper; otra sección del mismo paper da cifras distintas) | Detector YOLOv11x ajustado con pseudo-etiquetas · Deep-EIoU · GTA-Link · Re-ID OSNet [S2] |
| SoccerNet Team Ball Action Spotting 2025 (detectar acciones con balón y el equipo) | Kamil Czarnogorski | Team-mAP@1 60,03 | Reimplementación de T-DEED en el framework dude.k [S1] |
| SoccerNet 2026 Player-Centric Ball-Action Spotting (qué jugador hace qué acción y cuándo, 8 clases) | Paper de extensiones a las líneas base FOOTPASS | Macro F1 0,548 en test | TAAD, TAAD+GNN, TAAD+DST como líneas base [S3] |

**Conclusión de la referencia:** ningún ganador usa un modelo "de caja" sin ajustar. Todos ajustan el detector con video de fútbol y agregan re-identificación y refinamiento de trayectorias. El techo actual de reconstrucción completa desde TV está en torno a 64 de GS-HOTA, es decir, el problema no está resuelto ni para los mejores.

---

## 2. Componentes

### 2.1 Detección de jugadores, árbitros y balón

| Opción | Precisión publicada | Licencia | Estado |
|---|---|---|---|
| **RF-DETR N/S/M/L (Roboflow)** — recomendado | COCO AP: N 48,0 · S 52,9 · M 54,7 · L 56,5. SOTA entre detectores en tiempo real según su paper (ICLR 2026) [S4] | Apache 2.0 (tamaños XL y 2XL: PML 1.0) [S5] | Instalado y medido en CPU en este proyecto |
| D-FINE | COCO AP L 54,0 / X 55,8; con preentrenamiento Objects365: L 57,3 / X 59,3 [S6] | Apache 2.0, pero el repo advierte que los pesos con Objects365 no deben asumirse aprobados para uso comercial [S6] | No probado |
| DEIMv2 | COCO AP L 56,0 / X 57,8 [S7] | Licencia propia; uso comercial: contactar a los autores [S7] | No probado |
| YOLO11 / YOLO26 (Ultralytics) | COCO AP YOLO11 M 48,3 (tabla del paper de RF-DETR) [S4] | AGPL-3.0 o licencia Enterprise pagada; precio no publicado [S8] | No probado. Es lo que usó el ganador de SoccerTrack 2025 |
| YOLO-X | Usado por el ganador de SoccerNet GSR 2025 [S1] | No verificada en esta investigación | No probado |

**Punto crítico:** ninguno viene entrenado para fútbol. Hay que ajustarlo con video etiquetado de fútbol. Los videos de SoccerNet requieren firmar un acuerdo de confidencialidad (NDA), y su página no dice si permiten uso comercial [S9]. Esto debe confirmarse antes de entrenar un producto con ellos.

**Balón:** no encontré un modelo abierto, específico para el balón de fútbol, con licencia y resultados verificables. Se resuelve ajustando el detector elegido.

### 2.2 Seguimiento de todos los jugadores

| Opción | Precisión en fútbol | Licencia | Estado |
|---|---|---|---|
| Deep-EIoU | Núcleo de los ganadores de SoccerTrack 2025 y SoccerNet GSR 2025 [S1][S2] | **El repositorio no declara licencia** [S10]. Como principio general, sin licencia no hay permiso de uso: se necesita autorización de los autores (validar con abogado) | No usable hoy |
| **GTA-Link** (refinamiento de trayectorias) — recomendado | Parte del ganador de SoccerTrack 2025 [S2] | MIT [S11] | No probado |
| **BoT-SORT / OC-SORT / ByteTrack / McByte (paquete trackers)** — recomendado mientras no haya licencia de Deep-EIoU | ByteTrack: HOTA 0,42 en SoccerTrack 2025 (línea base) [S2] | Apache 2.0 [S12] | ByteTrack instalado y medido |
| **OSNet (torchreid)** para re-identificación — recomendado | Usado por ambos ganadores [S1][S2] | Código MIT [S13]; licencia de los pesos preentrenados no declarada | No probado |

### 2.3 Calibración de cancha (posiciones en metros)

| Opción | Precisión | Licencia | Estado |
|---|---|---|---|
| PnLCalib | Puntaje FS 67,6 en SN22-test-center (según el paper de SoccerMaster) [S14] | **GPL-2.0** [S15] | No probado |
| No Bells, Just Whistles | Sin cifras en el repositorio | **GPL-2.0** [S16] | No probado |
| SoccerMaster (modelo base específico de fútbol) | FS 75,8 vs 67,6 de PnLCalib en SN22-test-center [S14] | Código y modelo "serán publicados"; licencia no declarada [S14] | No disponible |
| **Construir un detector propio de puntos de la cancha con RF-DETR Keypoint** — recomendado si GPL no es aceptable | Método publicado por Roboflow: 32 puntos de la cancha, mínimo 4 visibles para calcular la posición [S17] | Apache 2.0 [S5] | Por construir |

**GPL:** obliga a publicar el código fuente bajo GPL cuando se distribuye el software. Si el sistema solo corre en tus servidores, el efecto es distinto. Esto debe validarlo un abogado antes de usar PnLCalib o No Bells, Just Whistles.

**Por tipo de cámara:** dron, gradería y TV muestran suficientes líneas; desde detrás del arco y desde la banca la calibración será parcial.

### 2.4 Pose 2D del cuerpo (perfilarse, controlar, saltar, driblar)

| Opción | Precisión publicada (COCO) | Licencia | Estado |
|---|---|---|---|
| **ViTPose++ Huge** — recomendado | 81,1 AP en COCO test-dev (ficha del modelo); 79,4 AP en COCO val (repositorio) [S18][S19] | **Apache 2.0** en la ficha oficial de Hugging Face; compatible con transformers [S18] | No probado; requiere GPU |
| RTMPose | 75,8 AP [S20] | Pesos sin documento oficial de licencia | **Descartado** por licencia no verificable |
| Sapiens (v1) | Grupo "81 %+ AP" [S20] | **CC BY-NC 4.0: prohíbe uso comercial** [S21] | **Descartado** |
| Sapiens2 (abril 2026, 308 puntos de cuerpo completo) | Sin cifras comparables publicadas [S22] | Licencia propia de Meta: no menciona uso comercial, prohíbe "procesamiento biométrico" [S23] | Riesgoso: requiere revisión legal |
| RF-DETR Keypoint | Recomendado por Roboflow; sin cifra de AP publicada [S24] | Apache 2.0 [S5] | Alternativa ligera |

**Limitación real:** los benchmarks de pose (COCO) usan personas grandes en la imagen. En tomas abiertas de fútbol el jugador ocupa pocos píxeles, y no encontré un benchmark de pose específico de fútbol. La precisión real se mide con clips propios; las tomas cercanas (dron bajo, gradería cercana) serán mucho mejores que las generales.

### 2.5 Pose 3D (ángulos reales de saltos, giros y perfiles)

| Opción | Qué entrega | Licencia | Estado |
|---|---|---|---|
| SAM 3D Body (Meta, nov-2025) | Malla 3D de cuerpo, pies y manos desde una sola imagen [S25] | SAM License: no declara explícitamente uso comercial [S25] | No probado; requiere revisión legal |

Esta es la parte menos investigada de este documento.

### 2.6 Seguir al jugador que el usuario marca

| Opción | Evidencia | Licencia | Estado |
|---|---|---|---|
| **SAM 2.1** — recomendado | Funciona con clic sobre el jugador; medido en este proyecto: siguió al objetivo en 20 de 20 cuadros | Apache 2.0 [S26] | Instalado; en CPU tardó 5,9 s por cuadro (inviable sin GPU) |
| SAM 3 (nov-2025) / SAM 3.1 (27-mar-2026) | Meta afirma mejora sobre SAM 2; no encontré la comparación numérica [S27] | SAM License, clasificada "Proprietary Free" por ScanCode; no declara explícitamente uso comercial [S28] | Requiere revisión legal |
| Clic sobre el ID del tracker + re-identificación OSNet | Técnica estándar en los ganadores [S1][S2] | Apache 2.0 / MIT | Opción de bajo costo |

### 2.7 Acciones (cabezazo, control, pase, tiro)

| Opción | Evidencia | Licencia | Estado |
|---|---|---|---|
| T-DEED / dude.k | Ganador SoccerNet TBAS 2025: Team-mAP@1 60,03 [S1] | No verificada | Pendiente |
| Líneas base FOOTPASS (TAAD) | Acción atribuida a un jugador, 8 clases; mejor resultado publicado: Macro F1 0,548 [S3] | No verificada; disponibilidad del código no confirmada | Pendiente |

### 2.8 Marco de referencia completo

| Opción | Qué es | Licencia |
|---|---|---|
| sn-gamestate (SoccerNet) | Pipeline completo de referencia: YOLOv11 + PRTReid + MMOCR (camiseta) + TVCalib / PnLCalib / No Bells, Just Whistles. Mantenido; última actualización 1-may-2026 [S29] | **GPL-3.0** [S29] |

---

## 3. Comprar en vez de construir

| Proveedor | Qué vende | Video de entrada | ¿Pose del cuerpo? | Precio |
|---|---|---|---|---|
| SkillCorner Data on Demand | Tracking XY, datos físicos e "inteligencia de juego" por API en 24-48 h [S30] | Tu propio video de una cámara táctica o abierta (juveniles, amistosos, torneos) [S30] | No mencionada [S30] | No publicado [S30] |
| Hawk-Eye SkeleTRACK (Sony) | Seguimiento esquelético, "decenas de puntos" por jugador [S31] | Arreglos de cámaras de alta velocidad en estadio [S31] | Sí | No publicado; infraestructura de liga; un club no puede pedirlo para su cancha de entrenamiento [S31] |
| TRACAB / Second Spectrum | Posiciones XY y balón [S31] | Cámaras instaladas en estadio [S31] | No mencionada [S31] | No publicado; nivel liga [S31] |
| Veo | Cámara + módulo Analytics (mapas 2D, mapas de calor, tiros, pases) [S32] | Cámara Veo | No | Precios en su página de precios (no consultada); exportación de datos crudos no mencionada [S32] |

**Hallazgo clave:** no encontré ningún proveedor que venda análisis de movimientos del cuerpo (pose) a partir de video amateur o de cualquier cámara. Las posiciones XY sí se pueden comprar (SkillCorner). El análisis biomecánico del 9 es lo que hay que construir, y es donde estaría la diferenciación del producto.

---

## 4. Infraestructura (GPU en la nube)

| Proveedor | GPU | Precio por hora |
|---|---|---|
| RunPod (pod a demanda) [S33] | L4 / RTX 4090 / L40S / A100 80 GB / H100 PCIe | USD 0,59 / 0,89 / 1,09 / 1,79 / 2,89 |
| Modal (cobro por segundo) [S34] | L4 / L40S / A100 80 GB / H100 | ≈ USD 0,80 / 1,95 / 2,50 / 3,95 (conversión propia: precio por segundo × 3.600) |

Modal Starter: USD 0 fijo + consumo, con USD 30 de crédito mensual [S34].

**No medido todavía:** la velocidad de cada componente en GPU. Sin esa medición no hay cálculo honesto de costo por minuto de video.

---

## 5. Stack recomendado hoy

| Función | Recomendado | Razón | Riesgo abierto |
|---|---|---|---|
| Detección | RF-DETR (N a L) ajustado con fútbol | Mejor precisión en tiempo real publicada + Apache 2.0 | Datos de entrenamiento con licencia comercial |
| Seguimiento | trackers (BoT-SORT/ByteTrack) + GTA-Link + OSNet | Mismos bloques que los ganadores, con licencias claras | Deep-EIoU, el mejor, no tiene licencia |
| Calibración | Detector propio de puntos de cancha (RF-DETR Keypoint) | Única vía con licencia limpia | Hay que construirlo; PnLCalib (GPL) es más rápido de usar si un abogado lo aprueba |
| Pose 2D | ViTPose++ Huge | Mejor precisión con licencia Apache verificada | Precisión en jugadores lejanos sin benchmark |
| Jugador marcado | SAM 2.1 en GPU + re-identificación OSNet | Probado en este proyecto, Apache 2.0 | SAM 3.1 podría ser mejor, licencia por revisar |
| Acciones | Pendiente | — | Licencias de T-DEED y FOOTPASS sin verificar |

## 6. Lo que este documento NO cubre o no pudo verificar
1. Comparación directa de detectores sobre video de fútbol (no hay una publicada que los incluya a todos).
2. Licencias de YOLO-X, T-DEED/dude.k y FOOTPASS.
3. Si SAM 3/3.1, Sapiens2 y SAM 3D Body permiten uso comercial: sus licencias no lo dicen explícitamente.
4. Si los videos de SoccerNet permiten uso comercial.
5. Velocidades en GPU.
6. Precios de Veo, SkillCorner, Ultralytics Enterprise y Hawk-Eye (no publicados o no consultados).
7. Pose 3D: investigación superficial.

---

## Fuentes
- [S1] SoccerNet 2025 Challenges Results — https://arxiv.org/pdf/2508.19182
- [S2] GTATrack: Winner Solution to SoccerTrack 2025 — https://arxiv.org/html/2602.00484
- [S3] SoccerNet 2026 Player-Centric Ball-Action Spotting — https://www.alphaxiv.org/abs/2606.09679
- [S4] RF-DETR paper — https://arxiv.org/html/2511.09554v2
- [S5] rfdetr en PyPI — https://pypi.org/project/rfdetr/
- [S6] D-FINE — https://github.com/Peterande/D-FINE
- [S7] DEIMv2 — https://github.com/Intellindust-AI-Lab/DEIMv2
- [S8] Ultralytics Docs — https://docs.ultralytics.com/
- [S9] SoccerNetv2 DevKit — https://github.com/SilvioGiancola/SoccerNetv2-DevKit
- [S10] Deep-EIoU — https://github.com/hsiangwei0903/Deep-EIoU
- [S11] GTA-Link — https://github.com/sjc042/gta-link
- [S12] trackers en PyPI — https://pypi.org/project/trackers/
- [S13] torchreid LICENSE — https://github.com/KaiyangZhou/deep-person-reid/blob/master/LICENSE
- [S14] SoccerMaster — https://arxiv.org/pdf/2512.11016
- [S15] PnLCalib — https://github.com/mguti97/PnLCalib
- [S16] No Bells, Just Whistles LICENSE — https://github.com/mguti97/No-Bells-Just-Whistles/blob/main/LICENSE
- [S17] Roboflow, calibración de cámara en deportes — https://roboflow.com/blog/camera-calibration-sports-computer-vision
- [S18] ViTPose++ Huge en Hugging Face — https://huggingface.co/usyd-community/vitpose-plus-huge
- [S19] ViTPose — https://github.com/ViTAE-Transformer/ViTPose
- [S20] CodeSOTA, keypoint detection — https://www.codesota.com/browse/computer-vision/keypoint-detection
- [S21] Sapiens LICENSE — https://github.com/facebookresearch/sapiens/blob/main/LICENSE
- [S22] Sapiens2 — https://github.com/facebookresearch/sapiens2
- [S23] Sapiens2 LICENSE — https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md
- [S24] Roboflow, mejores modelos de pose — https://blog.roboflow.com/best-pose-estimation-models/
- [S25] SAM 3D Body — https://github.com/facebookresearch/sam-3d-body
- [S26] SAM 2 — https://github.com/facebookresearch/segment-anything-2 · ficha SAM 2.1 — https://huggingface.co/facebook/sam2.1-hiera-large
- [S27] SAM 3 — https://github.com/facebookresearch/sam3
- [S28] ScanCode, SAM License 2025-11-19 — https://scancode-licensedb.aboutcode.org/sam-2025-11-19.yml
- [S29] sn-gamestate — https://github.com/SoccerNet/sn-gamestate
- [S30] SkillCorner Data on Demand — https://skillcorner.com/articles/data-on-demand
- [S31] GAMECODE.Ai, comparación Second Spectrum / TRACAB / Hawk-Eye (artículo de un competidor) — https://gamecode.ai/insights/articles/second-spectrum-vs-tracab-vs-hawk-eye/
- [S32] Veo, planes y add-ons — https://support.veo.com/hc/en-us/articles/4450464075025
- [S33] RunPod pricing — https://www.runpod.io/pricing
- [S34] Modal pricing — https://modal.com/pricing
