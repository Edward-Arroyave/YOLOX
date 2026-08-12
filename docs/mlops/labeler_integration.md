# Integración del etiquetador de casetes

## Implementación revisada

La interfaz existente está ubicada fuera de este repositorio en:

```text
D:\Projects\IA\vet\entrenamiento_yolox\proyectos\utils\cassette_labeler.py
```

Es una aplicación de escritorio basada en OpenCV. Actualmente:

- abre imágenes desde una carpeta local;
- crea y edita bounding boxes rectangulares;
- convierte rectángulos en polígonos;
- mueve, rota y edita vértices;
- usa las clases `cassette` y `test`;
- guarda una etiqueta YOLO TXT por imagen;
- genera `annotations.json` en formato COCO;
- conserva ancho y alto originales de las imágenes.

Ejemplo actual de ejecución:

```powershell
python cassette_labeler.py `
  --images D:\review\batch-2026-08\images `
  --output D:\review\batch-2026-08\labels `
  --json D:\review\batch-2026-08\review
```

## Brechas para el pipeline MLOps

La aplicación todavía no maneja:

- `batch_id` o `dataset_version`;
- estado individual `PENDING`, `APPROVED` o `REJECTED`;
- identidad del revisor y fecha de aprobación;
- descarga desde GCS o URLs firmadas;
- carga del resultado corregido a GCS;
- callback de revisión terminada hacia n8n;
- bloqueo de edición concurrente;
- validación final completa del COCO JSON;
- API HTTP ni documentación Scalar.

## Integración por etapas

### Etapa 1: aplicación local, sin API

1. n8n crea el paquete de revisión en GCS.
2. Un comando descarga `images/` y `prelabels/annotations.json`.
3. Se convierten las preetiquetas COCO a TXT YOLO antes de abrir la aplicación,
   porque actualmente carga anotaciones existentes desde TXT.
4. El revisor corrige y guarda.
5. Un comando `submit_review` valida y sube `annotations.json`.
6. `submit_review` llama al webhook documentado de n8n.

Esta etapa reutiliza la interfaz actual con cambios pequeños y permite validar
el proceso antes de construir un servicio web.

### Etapa 2: cliente conectado

La interfaz recibe `--batch-id` y una configuración de API, obtiene el paquete
de revisión, guarda avances y envía aprobación. La API debe implementar OpenAPI
y Scalar según `api_standards.md`.

### Etapa 3: revisión web multiusuario

Solo es necesaria si varios revisores trabajan simultáneamente. Requiere
autenticación, asignación de imágenes, bloqueo optimista, auditoría y control de
conflictos. No debe preceder a la validación de la Etapa 1.

## Archivos del paquete de revisión

```text
review-package/
├── manifest.json
├── images/
├── prelabels/
│   └── annotations.json
├── labels/                  # TXT que consume la UI actual
└── review/
    ├── annotations.json
    └── status.json
```

`status.json` propuesto:

```json
{
  "batch_id": "batch-2026-08",
  "status": "REVIEW_APPROVED",
  "reviewer": "reviewer@company.example",
  "reviewed_at": "2026-08-10T15:30:00Z",
  "approved_images": 1312,
  "rejected_images": 108,
  "annotations_sha256": "replace-with-real-checksum"
}
```

## Compatibilidad de anotaciones

El detector YOLOX utiliza `bbox` COCO en formato `[x, y, ancho, alto]`. Los
polígonos pueden conservarse en `segmentation`, pero el entrenamiento de
detección usa la caja envolvente calculada por el etiquetador.

Antes de aceptar el lote se debe comprobar que:

- cada `category_id` pertenece al catálogo aprobado;
- cada `file_name` existe exactamente una vez;
- IDs de imágenes y anotaciones son únicos;
- cajas y polígonos están dentro de la imagen;
- el área es positiva;
- el JSON final corresponde al mismo `batch_id` y manifiesto.

