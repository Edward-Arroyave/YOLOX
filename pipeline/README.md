# Pipeline de entrenamiento YOLOX

El pipeline se ejecuta por proyecto (`vet_yolox`, `lis_yolox` u otro perfil
configurado) y mantiene separados sus datasets y modelos.

## Flujo

1. Consulta `weights/<proyecto>/` en Azure.
2. Descarga el `best_ckpt.pth` de la última versión del proyecto.
3. Limpia el dataset local y descarga el lote configurado.
4. Entrena usando el último checkpoint como base de *fine-tuning*.
5. Exporta el nuevo mejor checkpoint a ONNX.
6. Calcula la siguiente versión y publica `.pth`, `.onnx` y `README.md`.
7. Elimina las imágenes y los pesos locales después de una publicación exitosa.

Si falla una etapa, las siguientes no se ejecutan y se conservan los archivos
locales para diagnóstico.

## Ejecución

Simulación para veterinaria:

```bash
python pipeline/run_training_pipeline.py \
  --project vet_yolox \
  --dry-run
```

Ejecución completa:

```bash
python pipeline/run_training_pipeline.py \
  --project vet_yolox \
  --yes-clean
```

Para LIS:

```bash
python pipeline/run_training_pipeline.py \
  --project lis_yolox \
  --yes-clean \
  --allow-no-base
```

`--allow-no-base` solo se utiliza para crear la primera versión `1.0.0`.
En ejecuciones posteriores, el pipeline exige y descarga la última versión.

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
