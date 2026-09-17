# Evaluación final del modelo

El pipeline descarga también el conjunto `test` del lote de Azure. La estructura es:

```text
test/images/<imágenes>
test/annotations/annotations.json
```

Las anotaciones son COCO, con los mismos identificadores y nombres de categoría que
training y val. Test se reserva para evaluar al terminar: no selecciona checkpoints.

Antes de exportar y publicar, el pipeline carga estrictamente el checkpoint base
seleccionado y el nuevo `best_ckpt.pth`, y evalúa ambos con las mismas condiciones.
Una arquitectura incompatible detiene la publicación; no se evalúan capas cargadas
parcialmente. Sin base se evalúa solo el nuevo. Sin carpeta test se registra el motivo;
con test presente pero inválido o con imágenes faltantes se detiene el pipeline.

`metrics.json`, en `test_evaluation`, y el HTML publicado incluyen mAP 50:95,
AP50, AP75, AP por tamaño, AR con 1/10/100 detecciones, AR por tamaño, AP/AR por
clase, latencia y FPS derivados de la medición del evaluador. La latencia incluye
forward y NMS, excluye carga de imágenes y no es una medición de producción ONNX.
Con un solo batch no hay muestra de tiempo y la latencia queda N/A.
AP/AR usan escala 0–1; un tamaño sin objetos evaluables queda N/A.
No se presentan AP como precision a un umbral fijo, ni la pérdida IoU como IoU medido.

Las diferencias son nuevo menos base. La comparación histórica de validación
permanece separada. Se verifican coincidencias de contenido de test con train/val
actuales y se advierte cuando existen; no se puede garantizar que el modelo base
no haya visto esas imágenes en otros entrenamientos históricos.

La sección guarda hashes de anotaciones y checkpoints, configuración y resultados
para auditar la comparación. Los hashes de imágenes están en `dataset.splits.test`.
