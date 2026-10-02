# Agente de Vision Label Studio

Permite lanzar el pipeline de entrenamiento desde la aplicación web
**Vision Label Studio** (botón "Entrenar"). La app no entra a esta VM:

1. deja una solicitud en Blob Storage,
   `<VLS_RUNS_CONTAINER>/<VLS_RUNS_PREFIX>/<runId>/request.json`;
2. enciende la máquina si estaba apagada.

Este agente corre como servicio en la VM, toma las solicitudes y ejecuta,
desde la raíz de este repositorio:

```
conda run -n ia-yolox-training python pipeline/run_training_pipeline.py \
  --prefix lis --dataset-folder <carpeta> --yes-clean [--base-checkpoint <archivo local>]
```

## Qué hace

- **Cola FIFO.** Atiende las solicitudes de la más antigua a la más reciente,
  una a la vez.
- **No pisa otros entrenamientos.** Si hay un `run_training_pipeline.py` o un
  `tools/train.py` lanzado por otra persona, deja la solicitud en `waiting`,
  con el motivo, y espera a que termine.
- **Modelo base.** Si en la app se eligió un `.pth`, lo descarga del
  contenedor de la solicitud a `.vls_runs/<runId>/base/` y lo pasa como
  `--base-checkpoint`. Si no, el pipeline usa la última versión publicada.
- **Progreso en vivo.** Define `VLS_PROGRESS_FILE`. El trainer escribe ahí
  `progress.json` (épocas, pérdidas, evaluaciones) y el agente lo sube, junto
  con las últimas 200 líneas del log, cada `VLS_LOG_UPLOAD_SECONDS`.
- **Resultado.** Al terminar escribe `status.json` (`succeeded` o `failed`)
  con la versión publicada, la duración y las mejores métricas.
- **Apagado.** Si la app encendió la máquina para esa solicitud y no queda
  cola ni otro entrenamiento, apaga la máquina (`sudo -n shutdown -h now`).

**Seguridad:** nada leído de Blob se ejecuta en una shell. Cada campo se
valida, la lista de argumentos se arma en el código y la carpeta de trabajo
tiene que ser este repositorio.

## Configuración

Usa el `.env` de este repositorio. `AZURE_STORAGE_CONNECTION_STRING` es la
misma conexión del pipeline. Variables opcionales:

| Variable | Por defecto | Uso |
|---|---|---|
| `VLS_RUNS_CONTAINER` | `yolox` | Contenedor de las solicitudes (`TRAINING_BLOB_CONTAINER` de la app) |
| `VLS_RUNS_PREFIX` | `vls-runs` | Carpeta de las solicitudes (`TRAINING_RUNS_PREFIX` de la app) |
| `VLS_CONDA_EXE` | `/opt/conda/bin/conda` | La fija `install.sh` |
| `VLS_POLL_SECONDS` | `20` | Cada cuánto busca solicitudes y revisa si la máquina está libre |
| `VLS_LOG_UPLOAD_SECONDS` | `30` | Cada cuánto sube el log y el progreso |

## Instalación en la VM (una vez, y tras cada actualización de Workbench)

```bash
cd /home/jupyter/2_VETERINARIA
git pull
bash pipeline/agent/install.sh            # entorno por defecto: ia-yolox-training
journalctl -u vls-training-agent -f       # debe decir "vigilando yolox/vls-runs…"
```

Si hay solicitudes en cola, el agente empieza enseguida. **Es un entrenamiento
real** (GPU).

Para probarlo a mano antes de instalarlo (Ctrl+C para salir):
```bash
conda run -n ia-yolox-training --no-capture-output python -m pipeline.agent.vls_agent
```

## Pruebas

```bash
python -m pytest tests/utils/test_vls_agent.py tests/utils/test_training_progress.py
```

No necesitan Azure, conda ni GPU: usan un almacén en memoria, un ejecutor
falso y un reloj falso.
