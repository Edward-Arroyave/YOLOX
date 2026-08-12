# Manual de operación mensual

## Antes del primer uso

1. Crear el bucket operacional de GCS y sus políticas de retención.
2. Crear el trabajo Azure -> GCS en Storage Transfer Service.
3. Configurar Pub/Sub para notificar el resultado de la transferencia.
4. Preparar la VM con GPU, SSD, drivers, entorno Python y este repositorio.
5. Configurar MLflow y comprobar que acepta métricas y artefactos.
6. Importar y activar los workflows de n8n.
7. Conectar la aplicación de etiquetado con los webhooks de revisión.
8. Ejecutar un lote pequeño de extremo a extremo sin promover el modelo.

## Lista de comprobación mensual

### Ingesta

- [ ] Se creó un único `batch_id` para el periodo.
- [ ] El manifiesto contiene blobs, ETags y tamaños.
- [ ] La transferencia terminó sin objetos fallidos.
- [ ] Los conteos de Azure y GCS coinciden.

### Revisión

- [ ] El modelo vigente generó las preetiquetas.
- [ ] Todas las imágenes quedaron aprobadas o rechazadas.
- [ ] Se registró la identidad del revisor.
- [ ] El JSON COCO corregido pasó validación.

### Entrenamiento

- [ ] El dataset tiene una versión única e inmutable.
- [ ] El conjunto de prueba no cambió.
- [ ] No hay imágenes duplicadas entre splits.
- [ ] MLflow registró dataset, commit, parámetros y checkpoint base.
- [ ] El entrenamiento terminó y produjo `best_ckpt.pth`.

### Evaluación y promoción

- [ ] Se compararon las métricas con producción.
- [ ] Se revisaron falsos negativos clínicamente importantes.
- [ ] Se comprobó paridad PyTorch/ONNX.
- [ ] Una persona autorizada aprobó o rechazó el candidato.
- [ ] La aplicación está usando la versión registrada como producción.

## Recuperación por etapa

| Estado | Acción segura |
|---|---|
| `TRANSFERRING` | Consultar el trabajo existente; no crear otro lote |
| `TRANSFERRED` | Revalidar y volver a solicitar preetiquetado |
| `PRELABELING` | Consultar el job del worker antes de repetir |
| `REVIEW_PENDING` | Reabrir el mismo lote en la aplicación |
| `TRAINING` | Consultar proceso y MLflow; reanudar desde checkpoint si aplica |
| `EVALUATING` | Repetir evaluación sin volver a entrenar |
| `PROMOTION_PENDING` | Mantener producción intacta hasta la decisión |
| `FAILED` | Corregir causa, registrar responsable y reanudar desde `failed_step` |

## Rollback de modelo

1. Identificar la última versión aprobada.
2. Cambiar atómicamente `registry/production.json` a esa versión.
3. Reiniciar o recargar la aplicación de inferencia.
4. Ejecutar una prueba de humo con imágenes conocidas.
5. Registrar motivo, responsable, hora y versiones involucradas.

Los modelos y datasets promovidos no se eliminan durante un rollback.

## Apagado de la VM

La VM se detiene únicamente cuando:

- los artefactos y logs están en GCS/MLflow;
- no existe un entrenamiento activo;
- la aplicación de revisión no depende de archivos exclusivamente locales; y
- el estado final del lote está persistido.

