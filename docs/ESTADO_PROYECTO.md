# PEAK — Estado del proyecto

Actualizado: 21-oct-2026. Proyecto propio de Andrés Escobar. No usa material de Hansi/TheNotCoach.
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
| GPU | Modal, plan Starter (USD 30 de crédito mensual; exige medio de pago registrado) | USD 0 dentro del crédito |

Detalle y fuentes: `COMPONENTES_CV_FUTBOL.md`. Costos de operación: `COSTOS_ESTIMADOS.md`.

## Plan de construcción
| Paso | Contenido | Estado |
|---|---|---|
| 0 | Repositorio GitHub conectado | Hecho (20-oct-2026) |
| 1 | Flujo básico con clip real: detección, seguimiento, clic en el 9, esqueleto, video de salida | Hecho (20-oct-2026). Ver resultados abajo |
| 2 | Calibración de cancha: mapa visto desde arriba con posiciones en metros | Hecho (21-oct-2026) con clip HD. Ver resultados abajo |
| 3 | Etiquetado en CVAT (jugadores, balón, puntos de cancha) | Pendiente |
| 4 | Entrenamiento con fútbol y comparación RF-DETR vs YOLO (GPU) | Pendiente |
| 5 | Seguimiento completo: OSNet, GTA-Link, SAM 3.1 | Pendiente |
| 6 | Medición de precisión contra etiquetas | Pendiente |

## Lo que Andrés debe entregar
- Paso 3: etiquetado.

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

## Resultado del paso 2 (21-oct-2026)
Calibración con PnLCalib (commit 8c87391, pesos SV, refinamiento PnL) en Modal, GPU T4.

### Clip de 1997 (TVN, 640×352): no sirve para calibrar
- Imágenes 530 a 866. PnLCalib calibró 26 de 337, y esas 26 están mal (el arco queda proyectado sobre los letreros).
- La integración está verificada: con la imagen de ejemplo oficial de PnLCalib la proyección calza exacta (`resultados/paso2_clip1997/integracion_ejemplo_oficial.jpg`).
- Probado en CPU, sin costo: pesos SV, SV_FT_WC14 y SV_FT_TSWC, con y sin refinamiento, y umbrales bajos (0,2 / 0,5). Todas las variantes fallan o calibran mal.
- Causa: en la imagen 290 el modelo detecta 4 puntos de cancha (umbral por defecto); en la imagen de ejemplo moderna, 20.
- Script del diagnóstico: `scripts/diagnostico/paso2_pnlcalib_cpu.py`.

### Clip Van Persie (España-Holanda, Mundial 2014, 1280×720)
- Jugada en vivo del gol: imágenes 0 a 238, cámara principal, sin cortes (lo demás del clip son repeticiones).
- Seguimiento igual al paso 1 (RF-DETR Large + Deep-EIoU propio + OSNet), ahora en GPU: 24 identidades.
- PnLCalib calibró 239 de 239. Se descartan 15 porque la posición estimada de la cámara se aleja más de 5 m de la mediana (la cámara de TV es fija): quedan 224 válidas.
- Posiciones dentro de la cancha (margen 2 m): 2.422 de 2.464. Las 42 restantes son todas del juez de línea (identidad 18).
- Ruido de posición frente a su promedio móvil de 0,5 s: mediana 0,18 m; percentil 90: 0,46 m.
- Velocidad suavizada a 0,5 s: mediana 3,7 m/s; percentil 90: 7,8 m/s; 54 de 893 mediciones sobre 10 m/s (cambios de identidad; lo corrige el paso 5).
- El 9 (Van Persie) es la identidad 24 desde la imagen 83 (verificado en la imagen 190, cabezazo). Antes de la 83 no está detectado de forma continua.
- Esqueleto ViTPose++ Huge del 9 en GPU (`modal_apps/paso2_esqueleto.py`): 156 imágenes (83 a 238), ninguna interpolada. El 86,1 % de los puntos del cuerpo supera confianza 0,3 (paso 1, clip de 1997: 50,7 %). Altura del 9 en la imagen: mediana 47 px (mín. 21, máx. 105). Puntos menos confiables: muñeca derecha 68,6 %, codo derecho 71,2 %. La confianza es la que reporta el modelo, no una precisión medida contra etiquetas (eso es el paso 6). Revisión visual en 8 imágenes, incluido el cabezazo: el esqueleto sigue el cuerpo. 0,125 s por imagen en T4.
- Tiempos en T4 por imagen: detección 0,054 s; apariencia 0,031 s; calibración 0,474 s.
- Código: `modal_apps/paso2_completo.py`, `scripts/paso2_mapa.py`. Resultados: `resultados/paso2_vanpersie/` (videos no se suben al repositorio).

### Consumo en Modal (21-oct-2026)
- Corrida clip 1997 (perdida por bloqueo de red, sin resultado), corrida clip 1997 completa (148,5 s medidos en GPU) corrida Van Persie (133,7 s medidos en GPU) y esqueleto Van Persie (19,5 s medidos en GPU; una corrida previa falló al inicio por una dependencia faltante).
- Precio usado: T4 USD 0,000164/s + 2 núcleos × USD 0,0000131/s + 16 GiB × USD 0,00000222/s = USD 0,00022572/s (modal.com/pricing, 21-oct-2026).
- No incluye arranque de contenedores ni construcción de imágenes. Total exacto: modal.com > Usage.

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
Dominios permitidos (claude.ai > Settings > Capabilities):
- Modelos (20-oct-2026): dl.fbaipublicfiles.com, huggingface.co, *.hf.co, download.openmmlab.com, storage.googleapis.com.
- Modal (21-oct-2026): api.modal.com, api.modal2.com, *.modal.com, e29e5794c3a8abd52b1549c7703408b1.r2.cloudflarestorage.com, modal-blobs.s3-accelerate.amazonaws.com, duot2hadvvkdh.cloudfront.net.

Modal desde este entorno:
- Instalar `modal[api-proxy-support]` (sin el paquete python-socks el cliente no atraviesa el proxy de red).
- Autenticación: `modal token new` entrega un enlace que Andrés aprueba en su navegador. El token vive solo en la sesión de chat donde se creó; en un chat nuevo hay que repetirlo.
- Volumen `peak-resultados` en Modal: copia de respaldo de cada corrida.
