# Documentación y gobierno del modelo

## Evidencias obligatorias por versión

Cada dataset y modelo debe poder reconstruirse sin depender de información oral
o de archivos temporales de la VM.

### Ficha del dataset

Debe registrar:

- versión y fecha;
- origen y periodo de captura;
- cantidad de imágenes por split y clase;
- criterios de inclusión y rechazo;
- responsables de revisión;
- método y semilla de partición;
- checksums de anotaciones y manifiestos;
- limitaciones, sesgos conocidos y datos sensibles;
- versión anterior de la que deriva.

### Ficha del modelo

Debe registrar:

- versión, commit Git y archivo de experimento;
- versión del dataset;
- checkpoint inicial;
- parámetros de entrenamiento;
- versiones de Python, CUDA, PyTorch y dependencias;
- métricas globales y por clase;
- comparación con producción;
- resultado de validación PyTorch/ONNX;
- limitaciones y casos de fallo conocidos;
- aprobador, fecha y decisión.

### Registro de ejecución

MLflow conserva parámetros, métricas y artefactos. El manifiesto del lote
conserva el estado del proceso. Ninguno sustituye la ficha de dataset o modelo;
los tres se enlazan usando `batch_id`, `dataset_version`, `run_id` y
`model_version`.

## Decisiones humanas

Estas decisiones no se automatizan:

- aceptar etiquetas corregidas como verdad de entrenamiento;
- aceptar excepciones a validaciones de datos;
- aprobar un candidato con degradación en alguna métrica;
- promover o retirar un modelo de producción;
- autorizar el uso de datos sensibles o nuevos orígenes.

## Historial de cambios

Cada modificación del pipeline debe actualizar como mínimo:

1. esta documentación cuando cambie el proceso;
2. `mlops/config/pipeline.example.yaml` cuando cambie la configuración;
3. los esquemas cuando cambie un contrato de datos;
4. el changelog o PR con impacto, migración y rollback;
5. las pruebas del componente modificado.

Toda API nueva o modificada debe actualizar también su contrato OpenAPI y la
referencia interactiva Scalar definida en `api_standards.md`.
