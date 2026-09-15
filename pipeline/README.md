# Pipeline de entrenamiento YOLOX

El pipeline recibe un prefijo (`vet` o `lis`) y deriva el nombre del proyecto y
del ONNX: `vet_yolox` o `lis_yolox`. Ambos usan la misma configuración `.env`.

## Flujo

1. Consulta las versiones bajo `weights/` en Azure.
2. Busca la carpeta de versión más alta y descarga su `best_ckpt.pth`.
3. Limpia el dataset local y descarga el lote configurado.
4. Entrena usando el último checkpoint como base de *fine-tuning*.
5. Exporta el nuevo mejor checkpoint a ONNX.
6. Calcula la siguiente versión y publica `.pth`, `.onnx`, `model_report.html` y `metrics.json`.
7. Elimina las imágenes y los pesos locales después de una publicación exitosa.

Si falla una etapa, las siguientes no se ejecutan y se conservan los archivos
locales para diagnóstico.

El orquestador configura `PYTHONPATH` para sus subprocesos, así que los comandos
de entrenamiento pueden importar `yolox` directamente desde el repositorio.
También valida las dependencias de entrenamiento y ONNX antes de eliminar o
descargar datasets.

## Ejecución

Simulación para veterinaria:

```bash
python pipeline/run_training_pipeline.py \
  --prefix vet \
  --dry-run
```

Ejecución completa:

```bash
python pipeline/run_training_pipeline.py \
  --prefix vet \
  --yes-clean
```

Por defecto se usa el mes actual en `PIPELINE_TIMEZONE` y se busca una carpeta
como `8-2026`. Para seleccionar otro lote:

```bash
python pipeline/run_training_pipeline.py \
  --prefix vet \
  --dataset-folder 7-2026 \
  --yes-clean
```

Un rango se procesa en orden, generando una versión por mes:

```bash
set -e
for lote in 6-2026 7-2026 8-2026; do
  python pipeline/run_training_pipeline.py \
    --prefix vet \
    --dataset-folder "$lote" \
    --yes-clean
done
```

Para LIS:

```bash
python pipeline/run_training_pipeline.py \
  --prefix lis \
  --yes-clean
```

El pipeline busca la carpeta de versión SemVer más alta en `weights/` que
contenga `best_ckpt.pth` y la descarga automáticamente. No se pasa la ruta del
checkpoint y el ONNX existente puede pertenecer a otro prefijo. Use
`--allow-no-base` únicamente cuando `weights/` todavía no contenga ningún
checkpoint.

La versión se calcula automáticamente. Si la última es `1.0.0`, el pipeline
publica `1.0.1`. `--version` es opcional y solo permite confirmar manualmente el
valor calculado; no permite saltar versiones.

Opciones de conservación:

```text
--skip-clean          conserva las imágenes locales
--keep-local-weights  conserva checkpoints locales después de publicar
```

## Resultado en Azure

```text
weights/1.0.1/
├── best_ckpt.pth
├── vet_yolox.onnx
└── model_report.html + metrics.json

<prefijo-lis>/1.0.0/
├── best_ckpt.pth
├── lis_yolox.onnx
└── model_report.html + metrics.json
```

La ficha t?cnica publicada registra proyecto, versión, modelo base, dataset,
experimento, GPU, batch, FP16, fecha, época, AP, tamaños y hashes SHA-256.

## Estructura fija del dataset

Todos los perfiles deben conservar estos nombres; no son variables de `.env`:

```text
<DATA_DIR>/training/images/
<DATA_DIR>/training/annotations/annotations.json
<DATA_DIR>/val/images/
<DATA_DIR>/val/annotations/annotations.json
```

La configuración compartida usa `PIPELINE_BLOB_BASE_PREFIX`,
`PIPELINE_WEIGHTS_PREFIX` y
`PIPELINE_EXP_FILE=exps/cassette/cassette_yolox.py`. Tanto `--prefix lis` como
`--prefix vet` cargan ese mismo experimento. El lote se recibe con
`--dataset-folder` o se calcula con el mes actual; su ruta local se deriva de
`AZURE_INGEST_DESTINATION`. La descripción de las variables está en
`.env.example` y en el README principal.

El batch del entrenamiento no es una variable de entorno. Se define una sola
vez mediante `TRAIN_BATCH_SIZE` en `exps/cassette/settings.py` y actualmente
vale `8`.
# Ficha técnica automática

El informe ahora es un HTML autónomo: `model_report.html`, con curvas vectoriales
integradas de calidad de validación, pérdida total, componentes de pérdida y
learning rate. Se abre sin instalar TensorBoard ni acceder a servicios externos;
también se puede imprimir desde el navegador. Sustituye al Markdown en nuevas
publicaciones. No necesita imágenes auxiliares ni archivos JavaScript.

`training_history` guarda los promedios de pérdidas por época y el learning rate
al cierre de cada época. `evaluations` conserva las evaluaciones reales; no se
interpolan métricas para épocas que no se evaluaron. Estos datos pertenecen a la
ejecución actual y no se mezclan con eventos antiguos del directorio TensorBoard.

Para convertir las métricas de una versión existente sin entrenar otra vez:

```bash
python tools/render_model_report.py --metrics /ruta/metrics.json --output /ruta/model_report.html
```

Los JSON antiguos permiten graficar las evaluaciones guardadas, pero no contienen
el historial completo de pérdidas: esas curvas aparecerán como no disponibles.
El comando no modifica ni publica los artefactos anteriores. La ficha HTML nueva
se publica automáticamente junto a los pesos en el siguiente entrenamiento.

Cada entrenamiento exitoso genera `metrics.json` y `model_report.html`. La captura
ocurre en el evaluador COCO existente y el documento se escribe después del
último ciclo de entrenamiento/evaluación, antes de exportar y publicar. No se
ejecuta otra inferencia ni otra evaluación. También funciona con `tools/train.py`.

El pipeline guarda cada ejecución en:

```text
YOLOX_outputs/<proyecto>/reports/<versión>/<id-ejecución>/
    metrics.json
    model_report.html
```

La limpieza de pesos conserva este historial. Azure recibe ambos archivos en
`<weights-prefix>/<versión>/`, junto al checkpoint y ONNX. No se genera ni publica un README por modelo. La publicación
usa `overwrite=False`; una subida fallida revierte los archivos subidos por esa
ejecución. El comando de publicación independiente requiere `--report-dir`.

## Procedencia y significado

- `best` contiene los resultados de la época que produjo `best_ckpt.pth`;
  `evaluations` conserva las evaluaciones de esta ejecución. La selección incluye
  el primer resultado cero para poder guardar un modelo inicial sin detecciones.
- AP50, AP75, mAP 50:95, AR (máximo 100 detecciones), AP/AR por clase y tiempo
  de inferencia proceden del evaluador. AP/AR usan escala 0–1. Las diferencias
  se expresan en unidades absolutas y puntos porcentuales.
- Las pérdidas son promedios por iteración de la última época, del proceso rank 0.
  No representan un promedio distribuido. Validation loss, precision, recall,
  IoU de detección y FPS son `N/A`: este pipeline no los calcula como métricas
  independientes. IoU loss no se presenta como IoU de detección.
- El experimento efectivo aporta arquitectura, dimensiones e hiperparámetros,
  incluyendo cambios recibidos por argumentos. Se registran hardware, fechas,
  checkpoint inicial, mejor época y hashes de artefactos.
- Los JSON COCO aportan imágenes y anotaciones por clase de cada split. Si test
  reutiliza las anotaciones de validación no se cuenta como conjunto independiente.
  El total es la suma de registros de los splits distintos; el crecimiento usa
  contenido único (SHA-256 de imágenes), por lo que duplicados cuentan una vez.
  El JSON conserva los hashes de imágenes y anotaciones para trazabilidad.
- El pipeline descarga `metrics.json` de la versión base si existe. Para modelos
  anteriores sin informe se puede reutilizar `curr_ap` del checkpoint cargado;
  las demás métricas y el crecimiento quedan como `N/A`. No se evalúa la base.
- La conclusión advierte si no se verifica la misma validación (anotaciones e
  imágenes). Una diferencia entre datasets distintos no demuestra superioridad.
- Los entrenamientos fallidos no generan una ficha de éxito. La escritura es
  exclusiva y las ejecuciones tienen identificadores únicos para no sobrescribir
  fichas anteriores. Un error de escritura detiene la publicación.

El esquema JSON versión 1 separa `model`, `training`, `hardware`, `dataset`,
`best`, `evaluations`, `losses`, `artifacts`, `base_model` y `comparison`.
Los valores ausentes se serializan como `null` y se muestran como `N/A`.
