# Workflows de n8n

## Propósito

n8n coordina estados y servicios. Las imágenes se transfieren directamente de
Azure Blob Storage a Google Cloud Storage; no atraviesan la memoria de n8n.

## Credenciales necesarias

- Credencial o webhook de Azure Event Grid.
- Identidad de GCP con permisos para ejecutar Storage Transfer Service.
- Permisos para iniciar, consultar y detener la VM de entrenamiento.
- Credencial para llamar al endpoint privado del worker.
- SMTP, Teams u otro canal para notificaciones y aprobaciones.

Los secretos no se escriben en nodos `Code`, manifiestos o parámetros visibles.

## Workflow `azure-image-intake`

Detonador: webhook invocado por Azure Event Grid para `BlobCreated`.

```text
Webhook
 -> responder validación de Event Grid si corresponde
 -> filtrar contenedor, prefijo y extensión
 -> construir idempotency_key
 -> insertar imagen pendiente
 -> responder HTTP 202
```

Clave recomendada:

```text
sha256(storage_account + container + blob_name + etag)
```

## Workflow `monthly-dataset-batch`

Detonador principal: `Schedule Trigger`, día 1 a las 02:00 en
`America/Bogota`. Debe existir también un webhook manual restringido para
repeticiones controladas.

```text
Schedule Trigger
 -> comprobar bloqueo mensual
 -> consultar objetos pendientes
 -> IF cantidad >= minimum_new_images
 -> crear batch_id y manifest.json
 -> marcar TRANSFERRING
 -> iniciar Storage Transfer Service
 -> guardar transfer_job_id
 -> liberar ejecución
```

Si no existen imágenes suficientes, se registra `SKIPPED_NO_DATA`. La ejecución
manual puede usar `force=true`, pero debe quedar auditada.

## Workflow `prelabel-and-review`

Detonador: mensaje Pub/Sub de transferencia completada, reenviado al webhook de
n8n. Durante la primera implementación se permite consultar periódicamente el
estado del trabajo.

```text
Webhook
 -> buscar batch por transfer_job_id
 -> ignorar evento ya procesado
 -> validar conteos/checksums
 -> iniciar VM
 -> esperar health check con timeout
 -> POST /jobs/prelabel al worker
 -> recibir callback PRELABELING_COMPLETE
 -> cambiar a REVIEW_PENDING
 -> notificar URL de revisión
```

## Workflow `review-to-training`

Detonador: webhook enviado por la aplicación de etiquetado.

Solicitud mínima:

```json
{
  "batch_id": "batch-2026-08",
  "status": "REVIEW_APPROVED",
  "annotations_uri": "gs://bucket/batches/batch-2026-08/review/annotations.json",
  "approved": 1312,
  "rejected": 108,
  "approved_by": "reviewer@company.example"
}
```

Pasos:

```text
Webhook
 -> verificar firma/autenticación
 -> validar estado REVIEW_PENDING
 -> validar COCO JSON
 -> crear versión inmutable del dataset
 -> marcar DATASET_READY
 -> POST /jobs/train al worker
 -> guardar run_id de MLflow
```

## Workflow `evaluate-and-promote`

Detonador: callback del worker al finalizar evaluación.

```text
Webhook
 -> verificar artefactos y métricas
 -> comparar candidato con producción
 -> IF quality gates aprobados
      -> PROMOTION_PENDING -> aprobación humana
    ELSE
      -> candidato rechazado -> notificar
 -> al aprobar: actualizar registry/production.json
 -> detener VM
```

Actualizar `production.json` debe ser la última operación y debe hacerse de
forma atómica. Un fallo previo nunca puede cambiar el modelo de producción.

## Reintentos

- Los webhooks pueden llegar repetidos; todos consultan el estado persistido.
- Las operaciones externas guardan su identificador antes de continuar.
- Se usa backoff exponencial para errores temporales.
- No se reintentan automáticamente errores de calidad o aprobación.
- Después del máximo de intentos, el lote pasa a `FAILED` y requiere decisión.

