# Estándar de API y documentación Scalar

## Regla obligatoria

Toda API HTTP creada para este proyecto debe cumplir simultáneamente:

1. contrato OpenAPI 3.1 disponible en `GET /openapi.json`;
2. documentación interactiva Scalar disponible en `GET /docs`;
3. endpoints funcionales bajo `/api/v1`;
4. autenticación en operaciones no públicas;
5. ejemplos de solicitud, respuesta y errores en OpenAPI;
6. pruebas que detecten diferencias entre implementación y contrato.

Una API no se considera terminada si Scalar no permite consultar todas sus
operaciones. En producción, `/docs` y `/openapi.json` permanecen protegidos por
autenticación de la empresa y nunca muestran secretos reales.

El contrato inicial del pipeline se encuentra en
`mlops/openapi/mlops-api.yaml`. Scalar debe leer el documento por URL, que es el
mecanismo recomendado para mantener la referencia sincronizada.

## Rutas reservadas

| Ruta | Uso |
|---|---|
| `/docs` | Scalar API Reference |
| `/openapi.json` | Contrato OpenAPI de la versión desplegada |
| `/health/live` | Proceso vivo |
| `/health/ready` | Dependencias listas |
| `/api/v1/...` | API estable versión 1 |

## Convenciones

- JSON usa `snake_case`.
- Fechas usan ISO 8601 UTC.
- IDs son opacos y no se reutilizan.
- Creaciones asíncronas responden `202 Accepted` y un `job_id`.
- Reintentos usan `Idempotency-Key`.
- Trazabilidad usa `X-Correlation-ID`.
- Errores usan un único esquema `ProblemDetails`.
- Cambios incompatibles requieren `/api/v2`; no se altera silenciosamente v1.

## Autenticación

Los endpoints operativos aceptan un token Bearer emitido para servicios. Los
webhooks además verifican firma o identidad del emisor. La documentación debe
describir el esquema, pero nunca incluir tokens SAS, credenciales de n8n o
claves de GCP.

## Ejemplo FastAPI + Scalar

Cuando se implemente el worker en FastAPI, la integración esperada es:

```python
from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

app = FastAPI(
    title="Cassette MLOps API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

@app.get("/docs", include_in_schema=False)
def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )
```

La dependencia debe fijarse al crear el servicio y actualizarse mediante una
revisión controlada; no se añade todavía al entorno de entrenamiento porque la
API aún no está implementada.

## Definition of Done de un endpoint

- [ ] Operación y modelos aparecen en OpenAPI.
- [ ] Scalar muestra descripción, seguridad y ejemplos.
- [ ] Se documentan códigos 2xx, 4xx y 5xx relevantes.
- [ ] Validación de entrada y autorización tienen pruebas.
- [ ] Reintentos no duplican trabajos.
- [ ] Logs incluyen `correlation_id`, no datos clínicos ni secretos.
- [ ] El consumidor n8n usa el contrato versionado.

