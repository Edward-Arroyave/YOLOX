# Arquitectura del pipeline MLOps mensual

## Objetivo

Automatizar la incorporación mensual de imágenes almacenadas en Azure Blob
Storage, preetiquetarlas con el modelo vigente, revisarlas en la aplicación de
etiquetado, entrenar YOLOX en una VM de GCP con NVIDIA T4 y publicar un modelo
nuevo solamente después de superar controles automáticos y humanos.

El modelo puede generar etiquetas preliminares automáticamente. Las etiquetas
de entrenamiento y la promoción a producción requieren aprobación humana.

## Responsabilidad de cada componente

| Componente | Responsabilidad |
|---|---|
| Azure Blob Storage | Fuente maestra de las imágenes procesadas |
| n8n | Programación, coordinación, estados, notificaciones y aprobaciones |
| GCP Storage Transfer Service | Copia masiva Azure Blob -> Google Cloud Storage |
| Google Cloud Storage | Área operacional, datasets versionados, ejecuciones y modelos |
| VM GCP + SSD + T4 | Descarga local, preetiquetado, validación, entrenamiento y exportación |
| Aplicación de etiquetado | Visualización y corrección de bounding boxes |
| MLflow | Parámetros, métricas, checkpoints y trazabilidad de experimentos |

n8n no debe transportar los binarios de las imágenes. Debe solicitar y vigilar
la transferencia, porque mover los archivos dentro de una ejecución de n8n
aumenta el uso de memoria, el tiempo de ejecución y el riesgo de reintentos
parciales.

## Almacenamiento

### Azure Blob Storage: fuente maestra

```text
cassette-images/
└── processed/
    └── YYYY/
        └── MM/
            ├── image-000001.png
            └── image-000002.png
```

Las imágenes no se mueven ni se borran al procesarlas. El lote registra el
nombre del blob, ETag, tamaño y checksum para evitar duplicados.

### Google Cloud Storage: área operacional

```text
gs://<mlops-bucket>/
├── landing/azure/YYYY/MM/                # réplica de Azure
├── batches/<batch_id>/
│   ├── manifest.json
│   ├── prelabels/annotations.json
│   ├── review/annotations.json
│   ├── review/status.json
│   ├── reports/data-validation.json
│   └── logs/
├── datasets/<dataset_version>/
│   ├── manifest.json
│   ├── annotations/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   ├── train/
│   ├── val/
│   └── test/
├── models/candidates/<model_version>/
│   ├── best_ckpt.pth
│   ├── model.onnx
│   ├── metrics.json
│   └── model-card.json
├── models/production/<model_version>/
└── registry/production.json              # puntero al modelo aprobado
```

`landing` puede aplicar una política de expiración después de confirmar que la
versión del dataset es reproducible. Los datasets y modelos aprobados no deben
sobrescribirse: cada versión es inmutable.

### SSD de la VM: espacio temporal de alto rendimiento

```text
/mnt/cassette-mlops/
├── cache/images/                         # descargas reutilizables
├── batches/<batch_id>/
├── datasets/<dataset_version>/
│   ├── annotations/
│   ├── train/
│   ├── val/
│   └── test/
├── runs/<run_id>/
└── models/
```

YOLOX siempre entrena desde este disco, no directamente desde Azure o GCS.

## Identificadores y estados

Ejemplos:

```text
batch_id:         batch-2026-08
dataset_version:  cassette-ds-2026-08-v1
run_id:           yolox-2026-08-01
model_version:    cassette-yolox-2026-08-01
```

Estados permitidos para un lote:

```text
CREATED
  -> TRANSFERRING
  -> TRANSFERRED
  -> PRELABELING
  -> REVIEW_PENDING
  -> REVIEW_APPROVED
  -> DATASET_READY
  -> TRAINING
  -> EVALUATING
  -> CANDIDATE_READY
  -> PROMOTION_PENDING
  -> PROMOTED
```

Un error lleva el lote a `FAILED`, conservando `failed_step`, mensaje y número
de reintentos. Cada paso comprueba el estado anterior y debe ser idempotente.

## Detonadores

### 1. Registro de imágenes nuevas

Azure Event Grid envía `BlobCreated` a un webhook de n8n:

```text
POST /webhook/azure-blob-created
```

Este flujo solo registra el blob en una tabla de control. No inicia un
entrenamiento por cada imagen.

Como alternativa inicial, el flujo mensual puede consultar Azure por prefijo y
fecha. Event Grid es preferible cuando el volumen crece.

### 2. Cierre mensual del lote

Un `Schedule Trigger` de n8n se ejecuta, por ejemplo, el día 1 a las 02:00 en
`America/Bogota`:

1. Adquiere el bloqueo `monthly-training-YYYY-MM`.
2. Selecciona blobs nuevos hasta el último día del mes anterior.
3. Si no hay suficientes imágenes, marca `SKIPPED_NO_DATA` y notifica.
4. Crea `batch_id` y `manifest.json`.
5. Ejecuta el trabajo de Storage Transfer Service Azure -> GCS.
6. Guarda el identificador del trabajo y termina la ejecución corta de n8n.

### 3. Transferencia completada

Storage Transfer Service publica su resultado en Pub/Sub. Un pequeño endpoint
HTTP (Cloud Run o Cloud Function) lo reenvía a:

```text
POST /webhook/gcs-transfer-complete
```

n8n valida el conteo y los checksums, cambia el lote a `TRANSFERRED`, enciende
la VM y solicita el trabajo de preetiquetado. Si no se configura Pub/Sub en la
primera versión, n8n puede consultar periódicamente el estado del trabajo.

### 4. Revisión terminada

La aplicación de etiquetado llama:

```text
POST /webhook/review-complete
```

Solo se acepta cuando todas las imágenes están `APPROVED` o `REJECTED`. Después
se construye una versión inmutable del dataset y se inicia el entrenamiento.

### 5. Entrenamiento terminado y promoción

El worker informa el resultado a:

```text
POST /webhook/training-complete
```

Si el candidato supera los umbrales, n8n solicita aprobación. La aprobación
llama a `/webhook/promote-model`; entonces se copia el artefacto candidato y se
actualiza atómicamente `registry/production.json`.

## Flujos de n8n

### Workflow A: `azure-image-intake`

```text
Webhook BlobCreated
 -> validar contenedor/prefijo/extensión
 -> calcular clave idempotente (account + container + blob + ETag)
 -> insertar si no existe
 -> responder 202
```

### Workflow B: `monthly-dataset-batch`

```text
Schedule Trigger / ejecución manual
 -> adquirir bloqueo
 -> consultar imágenes pendientes
 -> crear manifiesto
 -> iniciar Storage Transfer Service
 -> guardar transfer_job_id
 -> notificar "transferencia iniciada"
```

### Workflow C: `prelabel-and-review`

```text
Webhook transferencia completa
 -> validar transferencia
 -> iniciar VM
 -> esperar health check
 -> solicitar PRELABEL
 -> esperar callback del worker
 -> publicar lote en aplicación
 -> notificar al revisor
```

### Workflow D: `review-to-training`

```text
Webhook revisión completa
 -> validar COCO JSON y archivos
 -> crear split reproducible
 -> crear dataset versionado
 -> solicitar TRAIN a la VM
 -> registrar run_id de MLflow
```

### Workflow E: `evaluate-and-promote`

```text
Webhook entrenamiento completo
 -> comparar candidato contra producción
 -> verificar reglas de calidad
 -> solicitar aprobación humana
 -> promover o rechazar
 -> apagar VM
 -> notificar resultado
```

## Contrato con la aplicación de etiquetado

El trabajo de preetiquetado publica un COCO JSON con:

```json
{
  "images": [{"id": 1, "file_name": "image-000001.png", "width": 640, "height": 480}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 80], "area": 8000, "iscrowd": 0}],
  "categories": [{"id": 1, "name": "cassette"}, {"id": 2, "name": "test"}]
}
```

`bbox` usa el formato COCO `[x, y, ancho, alto]`. La aplicación debe conservar
IDs, registrar cambios y generar el JSON corregido en
`batches/<batch_id>/review/annotations.json`.

La interfaz actual fue revisada desde el proyecto externo
`D:\Projects\IA\vet\entrenamiento_yolox\proyectos\utils\cassette_labeler.py`.
Su integración progresiva está definida en `labeler_integration.md`.

## Construcción del dataset

El dataset mensual no contiene únicamente imágenes nuevas:

```text
dataset aprobado anterior + nuevas imágenes aprobadas
```

El conjunto de prueba permanece fijo para comparar versiones. Las nuevas
imágenes se asignan de forma reproducible a entrenamiento o validación. Cuando
existan varias fotos del mismo casete, paciente, dispositivo o sesión deben
permanecer en el mismo split para evitar fuga de información.

La estructura local resultante es compatible con la configuración actual:

```python
self.data_dir = "/mnt/cassette-mlops/datasets/cassette-ds-2026-08-v1"
self.train_image_dir = "train"
self.val_image_dir = "val"
self.test_image_dir = "test"
self.train_ann = "train.json"
self.val_ann = "val.json"
self.test_ann = "test.json"
```

## Comandos del worker

Descarga operacional:

```bash
gcloud storage rsync \
  gs://<mlops-bucket>/datasets/<dataset_version> \
  /mnt/cassette-mlops/datasets/<dataset_version> \
  --recursive
```

Entrenamiento:

```bash
python tools/train.py \
  -f exps/vet/vet_yolox.py \
  -d 1 -b 8 --fp16 --cache disk -l mlflow \
  data_dir /mnt/cassette-mlops/datasets/<dataset_version> \
  train_image_dir train val_image_dir val test_image_dir test \
  train_ann train.json val_ann val.json test_ann test.json
```

Los valores reales de batch size y caché deben ajustarse midiendo memoria RAM,
VRAM y velocidad del SSD de la VM.

## Reglas mínimas de calidad

Antes del entrenamiento:

- COCO JSON válido y sin IDs duplicados.
- Todas las imágenes referenciadas existen y se pueden decodificar.
- Bounding boxes con área positiva y dentro de la imagen.
- Categorías exactamente iguales a las configuradas en el experimento.
- Sin duplicados entre train, val y test.
- Cantidad mínima de ejemplos por clase.

Antes de promover:

- El candidato no reduce AP50:95 frente a producción más allá de la tolerancia.
- Recall por clase supera el mínimo empresarial.
- Falsos negativos críticos pasan revisión visual.
- El ONNX produce resultados equivalentes al checkpoint PyTorch.
- Dataset, código, parámetros y artefactos están vinculados en MLflow.

Los umbrales se definen en `mlops/config/pipeline.example.yaml`; inicialmente
deben bloquear la promoción automática y exigir aprobación humana.

## Seguridad

- Guardar SAS, tokens y claves únicamente en credenciales de n8n o Secret
  Manager; nunca en el repositorio, manifiestos o URLs registradas.
- Usar identidades con privilegios mínimos.
- Cifrar almacenamiento y comunicaciones.
- No registrar imágenes ni datos sensibles en logs.
- Para imágenes clínicas, verificar anonimización, residencia, retención y los
  acuerdos aplicables antes de transferir datos entre Azure y GCP.
