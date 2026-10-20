# Costo estimado de procesamiento — desde benchmarks publicados (sin gastar)

Fecha: 20-oct-2026. Base: video de 25 imágenes por segundo, por lo que 1 minuto = 60 × 25 = 1.500 imágenes.

> "Si no te puedo descomponer este número en partes que puedas revisar una por una, no es un número: es una impresión, y debes tratarla como tal."

## Costo por minuto de video (peor caso: calibración y SAM en cada imagen)

| Pieza | Velocidad publicada (GPU del benchmark) | Segundos de GPU por minuto de video | Precio GPU usado | USD por minuto de video |
|---|---|---|---|---|
| Detección RF-DETR M | 4,4 ms por imagen en T4 [1] | 1.500 × 0,0044 = 6,6 | T4 Modal USD 0,000164/s [2] | 6,6 × 0,000164 = 0,0011 |
| Seguimiento Deep-EIoU (incluye su propio detector + re-ID) | 14,6 imágenes/s en RTX 4080 [3] | 1.500 ÷ 14,6 = 102,7 | RTX 4090 RunPod USD 0,89/h (la 4080 no está listada) [4] | 102,7 × 0,89 ÷ 3.600 = 0,0254 |
| Esqueleto ViTPose-H | 241 imágenes/s en A100, lotes de 64 [5] | 1.500 ÷ 241 = 6,2 | A100 80 GB RunPod USD 1,79/h [4] | 6,2 × 1,79 ÷ 3.600 = 0,0031 |
| Calibración PnLCalib | 4 imágenes/s en RTX 2080 Ti [6] | 1.500 ÷ 4 = 375 | L4 RunPod USD 0,59/h, la más barata listada (2080 Ti no listada) [4] | 375 × 0,59 ÷ 3.600 = 0,0615 |
| Recuperar al 9: SAM 2.1 large | 39,5 imágenes/s en A100 [7] | 1.500 ÷ 39,5 = 38,0 | A100 80 GB RunPod USD 1,79/h [4] | 38,0 × 1,79 ÷ 3.600 = 0,0189 |
| **Total peor caso** | | **528,5 s (8,8 min)** | | **USD 0,1099** |

**Mejor caso** (cámara fija: se calibra una vez; SAM casi no se usa): 6,6 + 102,7 + 6,2 = 115,6 s de GPU (con decimales completos) y 0,0011 + 0,0254 + 0,0031 = **USD 0,0296 por minuto de video**.

Por video de 5 minutos: peor caso 5 × 0,1099 = **USD 0,55**; mejor caso 5 × 0,0296 = **USD 0,15**.

## Almacenamiento

- Tamaño de un video de 5 minutos a 1080p y 25 imágenes/s, con la tasa recomendada por YouTube (8 Mbps) [8]: 8 Mbps × 300 s = 2.400 megabits = 300 MB = 0,3 GB.
- Cloudflare R2: USD 0,015 por GB al mes, sin costo de descarga; 10 GB gratis al mes [9].
- Costo por video guardado: 0,3 × 0,015 = USD 0,0045 al mes.

## Ejemplo: 200 videos de 5 minutos al día (escenario que planteaste en tu conversación con Gemini)

| Rubro | Cálculo | USD al mes |
|---|---|---|
| Procesamiento, peor caso | 200 videos × 5 min × 30 días × 0,1099 | 3.298 |
| Procesamiento, mejor caso | 200 videos × 5 min × 30 días × 0,0296 | 887 |
| Almacenamiento de un mes de videos | 200 × 30 × 0,3 GB × 0,015 | 27 (se acumula cada mes que se guarden) |

## Qué hace que el costo real sea MAYOR que esta estimación
1. Los benchmarks usan condiciones ideales: TensorRT en RF-DETR, lotes de 64 en ViTPose. Procesar un solo jugador por imagen es más lento.
2. No incluye decodificar el video, subirlo, ni el tiempo en que la GPU queda encendida sin trabajar.
3. Las GPU del benchmark no siempre son las que se arriendan; se usó el precio de la más cercana o la más barata listada.
4. No incluye la etapa de machine learning (rúbrica y plan) ni el servidor web.

Ese factor de diferencia solo se conoce midiendo.

## Fuentes
- [1] RF-DETR paper — https://arxiv.org/html/2511.09554v2
- [2] Modal pricing — https://modal.com/pricing
- [3] Deep-EIoU paper — https://ar5iv.labs.arxiv.org/html/2306.13074
- [4] RunPod pricing — https://www.runpod.io/pricing
- [5] ViTPose paper, tabla 9 — https://arxiv.org/pdf/2204.12484v3
- [6] PnLCalib paper — https://arxiv.org/pdf/2404.08401v3
- [7] SAM 2 repositorio — https://github.com/facebookresearch/sam2
- [8] YouTube, tasas de bits recomendadas — https://support.google.com/youtube/answer/1722171
- [9] Cloudflare R2 pricing — https://developers.cloudflare.com/r2/pricing/
