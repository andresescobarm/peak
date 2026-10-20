# PEAK — Estado del proyecto

Actualizado: 20-oct-2026. Proyecto propio de Andrés Escobar. No usa material de Hansi/TheNotCoach.
Repositorio: github.com/andresescobarm/peak (privado). Copia de los documentos en el proyecto "Futbol y Millones" de claude.ai, carpeta `proyecto9`.

## Objetivo
Sistema que analiza video de un futbolista target (primero: 9 / centrodelantero), en relación con compañeros y rivales y en sus movimientos finos (perfilarse, controlar, saltar, driblar, definir). Entrega una rúbrica de errores y debilidades, y un plan de entrenamiento específico. Usuarios: nueves profesionales y en formación. Luego se amplía a otras posiciones.

## Decisiones del dueño
- Videos: banca, detrás del arco, gradería, dron y transmisión de TV.
- El usuario marca al jugador target al inicio.
- Rúbrica: desde literatura especializada y un pool de nueves históricos agrupados por fortalezas.
- Etiquetado: lo hace Andrés. Ejemplos de ejecuciones malas o promedio: se buscarán.
- Motor de machine learning propio que aprende del pool etiquetado.
- Criterio de selección de tecnología: lo mejor disponible para fútbol. Las licencias no son un filtro; se informa su costo.
- Tracker: implementación propia del método Deep-EIoU, escrita desde su paper.
- Costos: se estiman investigando, sin gastar.

## Stack definitivo de visión computacional
| Etapa | Herramienta | Costo |
|---|---|---|
| Detección | RF-DETR ajustado con fútbol (se compara contra YOLO con clips propios; si gana YOLO, se usa con licencia Enterprise) | USD 0 (YOLO Enterprise: precio no publicado) |
| Seguimiento | Deep-EIoU (implementación propia) + OSNet + GTA-Link | USD 0 |
| Recuperar al 9 | SAM 3.1 | USD 0 |
| Calibración de cancha | PnLCalib | USD 0 |
| Esqueleto | ViTPose++ | USD 0 |
| Etiquetado | CVAT (instalado en servidor propio) | USD 0 |
| GPU | Modal (USD 30 de crédito mensual gratis) | USD 0 dentro del crédito |

Detalle y fuentes: `COMPONENTES_CV_FUTBOL.md`. Costos de operación: `COSTOS_ESTIMADOS.md`.

## Plan de construcción
| Paso | Contenido | Estado |
|---|---|---|
| 0 | Repositorio GitHub conectado | Hecho (20-oct-2026) |
| 1 | Flujo básico con clip real: detección, seguimiento, clic en el 9, esqueleto, video de salida | Hecho (20-oct-2026). Ver resultados abajo |
| 2 | Calibración de cancha: mapa visto desde arriba con posiciones en metros | Pendiente |
| 3 | Etiquetado en CVAT (jugadores, balón, puntos de cancha) | Pendiente |
| 4 | Entrenamiento con fútbol y comparación RF-DETR vs YOLO (GPU) | Pendiente |
| 5 | Seguimiento completo: OSNet, GTA-Link, SAM 3.1 | Pendiente |
| 6 | Medición de precisión contra etiquetas | Pendiente |

## Lo que Andrés debe entregar
- Paso 3: etiquetado.
- Paso 4: cuenta de Modal.

## Resultado del paso 1 (20-oct-2026)
Clip: compilación de TV (TVN), 640×352, 29,97 imágenes/s. Jugada usada: imágenes 530 a 866 (337 imágenes, sin cortes de cámara).
- Detección RF-DETR Large (sin ajustar con fútbol) y seguimiento Deep-EIoU propio + OSNet (pesos MSMT17, licencia MIT).
- 21 identidades para menos jugadores reales: el seguimiento asigna números nuevos tras pérdidas, sobre todo en el giro final de cámara. Lo corrige el paso 5.
- El 9 tuvo dos números: 16 hasta la imagen 795 y 21 desde la 796 (intercambio al chocar con un defensor, verificado imagen por imagen).
- El 9 aparece en 255 imágenes; en 13 su posición es interpolada (el detector no lo vio) y ahí no se calcula esqueleto.
- Esqueleto ViTPose++ Huge en 242 imágenes. Solo el 50,7 % de los puntos del cuerpo supera confianza 0,3: en la imagen 700 el 9 mide 25 × 35 píxeles. Los puntos menos confiables son las muñecas (24 % y 26 % sobre 0,3) y las caderas (36 %). Estos porcentajes corresponden a la primera corrida de 255 imágenes.
- Tiempos en CPU por imagen: detección 0,81 s; apariencia 8,28 s (ya reducido en el código: solo detecciones de confianza alta); esqueleto 2,08 s.
- Conclusión: el flujo funciona de punta a punta con fútbol real. Para medir movimientos finos con confianza se necesitan videos de mayor resolución o tomas más cercanas.
- Código: `src/peak/tracking/deep_eiou.py`, `scripts/paso1_seguimiento.py`, `scripts/paso1_esqueleto.py`. Resultados: `resultados/paso1/`.

## Historial: prueba de velocidad en CPU (20-oct-2026)
Herramientas del primer borrador (reemplazadas por el stack definitivo). Entorno: 2 núcleos de CPU, 7,8 GB RAM, sin GPU. Video de ejemplo del repositorio oficial de SAM 2 (no es fútbol).

| Modelo | Segundos por cuadro |
|---|---|
| RF-DETR Nano / Small / Medium | 0,233 / 0,382 / 0,528 |
| ByteTrack | 0,013 |
| RTMPose-s / m / x | 0,013 / 0,029 / 0,193 |
| SAM 2.1 tiny / small | 5,902 / 6,375 |

Script y resultados: `benchmarks/fase1_cpu/`.

## Entorno
Dominios permitidos para descargar modelos (claude.ai > Settings > Capabilities, agregados el 20-oct-2026): dl.fbaipublicfiles.com, huggingface.co, *.hf.co, download.openmmlab.com, storage.googleapis.com.
