# Correcciones al pre-etiquetado del piloto (clip Van Persie, imágenes 0-238)

Base: salida del sistema (`resultados/paso2_vanpersie/tracks.json`, `balls.json`).
Correcciones hechas antes de entregarlo a CVAT, cada una verificada imagen por imagen:

1. Van Persie: la identidad 16 lo sigue en 6-29 y 34-47; en 30-33 y desde 48 salta al defensor de blanco.
   Van Persie queda como una sola trayectoria: 16 (6-29, 34-47) + 15@71 (el recuadro estaba sobre él) + 24 (83-238),
   más una caja aproximada en la imagen 0 para que Andrés la ajuste. El defensor queda como trayectoria aparte (30-33, 48-170).
2. Identidad 6: en 44-86 la interpolación cruza de un jugador de azul a uno de blanco. Se corta: 0-43 y 87-88 (nueva).
3. Identidad 10: en 129-171 el recuadro queda sobre pasto (el jugador salió de la imagen). Se corta: 0-128 y 172-174 (nueva).
4. Identidad 14: en 24-73 el recuadro interpolado queda sobre pasto. Se corta: 3-23 y 74-81 (nueva).
5. Balón 177-197: se eliminan detecciones fuera del arco (una en la tribuna y un objeto en el pasto); el balón ya está en la red.

Validado con datumaro (librería de formatos de CVAT): 239 imágenes, 38 trayectorias.
