# Pipeline de entrenamiento YOLOX

El pipeline recibe un prefijo (`vet` o `lis`) y deriva el nombre del proyecto y
del ONNX: `vet_yolox` o `lis_yolox`. Ambos usan la misma configuración `.env`.

## Flujo

1. Consulta las versiones bajo `weights/` en Azure.
2. Busca la carpeta de versión más alta y descarga su `best_ckpt.pth`.
3. Limpia el dataset local y descarga el lote configurado.
4. Entrena usando el último checkpoint como base de *fine-tuning*.
5. Exporta el nuevo mejor checkpoint a ONNX.
6. Calcula la siguiente versión y publica `.pth`, `.onnx` y `README.md`.
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
└── README.md

<prefijo-lis>/1.0.0/
├── best_ckpt.pth
├── lis_yolox.onnx
└── README.md
```

El README publicado registra proyecto, versión, modelo base, dataset,
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
`PIPELINE_WEIGHTS_PREFIX` y `PIPELINE_EXP_FILE_TEMPLATE`. El lote se recibe con
`--dataset-folder` o se calcula con el mes actual; su ruta local se deriva de
`AZURE_INGEST_DESTINATION`. La descripción de las variables está en
`.env.example` y en el README principal.
