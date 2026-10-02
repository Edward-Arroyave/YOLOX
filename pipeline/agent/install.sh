#!/usr/bin/env bash
# Instala (o reinstala) el agente de Vision Label Studio como servicio systemd.
# Idempotente: repítelo tras cada actualización de Workbench, que reemplaza el
# disco de arranque y borra /etc/systemd/system.
#
# Uso, desde la raíz del repositorio en la VM (como el usuario jupyter):
#   bash pipeline/agent/install.sh [entorno-conda]      # por defecto: ia-yolox-training
set -euo pipefail

ENV_NAME="${1:-ia-yolox-training}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVICE=vls-training-agent
RUN_USER="$(id -un)"

CONDA_EXE="${CONDA_EXE:-$(command -v conda || echo /opt/conda/bin/conda)}"
ENV_PREFIX="$("$CONDA_EXE" env list | awk -v env="$ENV_NAME" '$1 == env {print $NF}')"
if [ -z "$ENV_PREFIX" ]; then
  echo "No existe el entorno conda '$ENV_NAME'." >&2
  exit 1
fi
PYTHON="$ENV_PREFIX/bin/python"

"$PYTHON" -c "import azure.storage.blob, dotenv" \
  || { echo "Falta azure-storage-blob o python-dotenv en '$ENV_NAME' (pip install -r requirements.txt)." >&2; exit 1; }
[ -f "$REPO_DIR/.env" ] || { echo "Falta $REPO_DIR/.env (con AZURE_STORAGE_CONNECTION_STRING)." >&2; exit 1; }

sudo tee "/etc/systemd/system/$SERVICE.service" >/dev/null <<UNIT
[Unit]
Description=Vision Label Studio - agente de entrenamiento
After=network-online.target
Wants=network-online.target

[Service]
User=$RUN_USER
WorkingDirectory=$REPO_DIR
Environment=VLS_CONDA_EXE=$CONDA_EXE
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON -m pipeline.agent.vls_agent
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE"
sudo systemctl restart "$SERVICE"
echo "Servicio $SERVICE instalado desde $REPO_DIR con $PYTHON."
echo "Ver: systemctl status $SERVICE   |   journalctl -u $SERVICE -f"
sudo -n true 2>/dev/null && echo "sudo sin contraseña: OK (el agente podrá apagar la máquina)." \
  || echo "AVISO: sin sudo sin contraseña el agente no podrá apagar la máquina al terminar."
