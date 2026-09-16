#!/usr/bin/env python3
"""Pipeline segmentado: modelo base, datos, entrenamiento y publicación."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from exps.cassette.settings import TRAIN_BATCH_SIZE  # noqa: E402
from tools.ingest_blob_storage import (  # noqa: E402
    combine_prefix,
    create_container_client,
    load_environment,
)
from tools.publish_weights import (  # noqa: E402
    VERSION_PATTERN,
    format_version,
    next_patch_version,
    normalize_prefix as normalize_weights_prefix,
    version_number,
)


PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
RUNTIME_DEPENDENCIES = {
    "torch": "torch",
    "thop": "thop",
    "cv2": "opencv-python",
    "pycocotools": "pycocotools",
    "tensorboard": "tensorboard",
    "onnx": "onnx",
}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} debe ser true o false")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Falta {name} en el entorno o .env")
    return value


def normalize_model_prefix(prefix: str) -> str:
    normalized = prefix.strip().lower()
    if not PREFIX_PATTERN.fullmatch(normalized):
        raise ValueError("--prefix solo admite letras minúsculas, números, _ y -")
    return normalized


def experiment_path(configured: str) -> Path:
    return resolve_repo_path(configured)


def current_dataset_folder(timezone_name: str) -> str:
    try:
        current = datetime.now(ZoneInfo(timezone_name))
    except Exception as exc:
        raise ValueError(f"PIPELINE_TIMEZONE no es válida: {timezone_name}") from exc
    return f"{current.month}-{current.year}"


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def display_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def add_repository_to_pythonpath(environment: dict[str, str]) -> None:
    repository = str(REPOSITORY_ROOT)
    current = environment.get("PYTHONPATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if repository not in entries:
        entries.insert(0, repository)
    environment["PYTHONPATH"] = os.pathsep.join(entries)


def missing_runtime_dependencies(require_onnxsim: bool) -> list[str]:
    dependencies = dict(RUNTIME_DEPENDENCIES)
    if require_onnxsim:
        dependencies["onnxsim"] = "onnx-simplifier"
    return [
        package
        for module, package in dependencies.items()
        if importlib.util.find_spec(module) is None
    ]


def run_stage(
    name: str,
    command: list[str],
    dry_run: bool,
    environment: dict[str, str] | None = None,
) -> None:
    print(f"\n=== {name} ===", flush=True)
    print(display_command(command), flush=True)
    if not dry_run:
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
        )


def find_latest_version(blob_names: list[str], weights_prefix: str) -> str | None:
    base = weights_prefix.rstrip("/") + "/"
    versions = []
    for name in blob_names:
        if not name.startswith(base) or not name.endswith("/best_ckpt.pth"):
            continue
        relative = name[len(base):]
        version = relative.split("/", 1)[0]
        match = VERSION_PATTERN.fullmatch(version)
        if match:
            versions.append(tuple(int(part) for part in match.groups()))
    return format_version(max(versions)) if versions else None


def download_base_checkpoint(
    client,
    weights_prefix: str,
    version: str,
    destination: Path,
) -> None:
    blob_name = combine_prefix(combine_prefix(weights_prefix, version), "best_ckpt.pth")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with temporary.open("wb") as output:
            client.get_blob_client(blob_name).download_blob().readinto(output)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"El modelo base está vacío: {blob_name}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def clean_local_weights(output_project: Path, artifacts_project: Path) -> int:
    deleted = 0
    if output_project.is_dir():
        for pattern in ("*.pth", "*.pt", "*.onnx"):
            for item in output_project.rglob(pattern):
                if item.is_file():
                    item.unlink()
                    deleted += 1
    if artifacts_project.is_dir():
        deleted += sum(1 for item in artifacts_project.rglob("*") if item.is_file())
        shutil.rmtree(artifacts_project)
    return deleted


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline completo por proyecto.")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefijo del modelo, por ejemplo vet o lis.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "Versión SemVer opcional. Por defecto aumenta automáticamente "
            "el parche de la última versión."
        ),
    )
    parser.add_argument(
        "--dataset-folder",
        default=None,
        help=(
            "Lote remoto, por ejemplo 8-2026. Si se omite, usa el mes actual "
            "en PIPELINE_TIMEZONE."
        ),
    )
    parser.add_argument("--env-file", default=None, help="Archivo .env alternativo.")
    parser.add_argument(
        "--base-checkpoint", default=None,
        help="Checkpoint local inicial; reemplaza la última versión publicada como base.",
    )
    parser.add_argument(
        "--yes-clean", action="store_true", help="Autoriza limpiar imágenes locales."
    )
    parser.add_argument(
        "--skip-clean", action="store_true", help="Conserva las imágenes locales."
    )
    parser.add_argument(
        "--keep-local-weights",
        action="store_true",
        help="Conserva checkpoints locales después de publicar.",
    )
    parser.add_argument(
        "--allow-no-base",
        action="store_true",
        help="Permite publicar el primer modelo cuando weights/ no tiene un checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra el flujo sin descargar, borrar, entrenar ni publicar.",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        loaded = load_environment(args.env_file)
        if loaded:
            print(f"Configuración cargada desde: {loaded}")
        if args.yes_clean and args.skip_clean:
            raise ValueError("No combine --yes-clean con --skip-clean")
        if not args.dry_run and not args.yes_clean and not args.skip_clean:
            raise ValueError("Use --yes-clean para limpiar o --skip-clean para conservar datos")

        extra_train_args = shlex.split(os.getenv("PIPELINE_TRAIN_ARGS", ""))
        for token in extra_train_args:
            option = token.split("=", 1)[0]
            if (option.startswith("-c") and not option.startswith("--")) or (
                option.startswith("--") and len(option) > 2
                and any(name.startswith(option) for name in ("--ckpt", "--resume"))
            ):
                raise ValueError(
                    "PIPELINE_TRAIN_ARGS no admite --ckpt/-c ni --resume; "
                    "use --base-checkpoint para registrar la base real"
                )
        local_base = resolve_repo_path(args.base_checkpoint) if args.base_checkpoint else None
        if local_base and (not local_base.is_file() or local_base.stat().st_size == 0):
            raise ValueError(f"El checkpoint base no existe o está vacío: {local_base}")

        prefix = normalize_model_prefix(args.prefix)
        project = f"{prefix}_yolox"
        timezone_name = os.getenv("PIPELINE_TIMEZONE", "America/Bogota").strip()
        dataset_folder = (
            args.dataset_folder.strip()
            if args.dataset_folder
            else current_dataset_folder(timezone_name)
        )
        if not dataset_folder or "/" in dataset_folder or "\\" in dataset_folder:
            raise ValueError("--dataset-folder debe ser el nombre de un lote, no una ruta")
        blob_base_prefix = required_env("PIPELINE_BLOB_BASE_PREFIX")
        weights_prefix = normalize_weights_prefix(required_env("PIPELINE_WEIGHTS_PREFIX"))
        exp_file = experiment_path(
            os.getenv("PIPELINE_EXP_FILE", "exps/cassette/cassette_yolox.py")
        )
        data_dir = resolve_repo_path(
            os.getenv("AZURE_INGEST_DESTINATION", "datasets")
        ) / dataset_folder
        output_dir = resolve_repo_path(os.getenv("PIPELINE_OUTPUT_DIR", "YOLOX_outputs"))
        output_project = output_dir / project
        checkpoint = output_project / "best_ckpt.pth"
        artifacts_project = resolve_repo_path(
            os.getenv("PIPELINE_ARTIFACTS_DIR", ".pipeline_artifacts")
        ) / project
        devices = int(os.getenv("PIPELINE_DEVICES", "1"))
        batch_size = TRAIN_BATCH_SIZE
        fp16 = env_bool("PIPELINE_FP16", True)
        no_onnx_simplify = env_bool("PIPELINE_ONNX_NO_SIMPLIFY", False)
        if devices < 1 or batch_size < 1:
            raise ValueError("PIPELINE_DEVICES y TRAIN_BATCH_SIZE deben ser mayores que cero")
        if not exp_file.is_file():
            raise ValueError(f"No existe el experimento configurado: {exp_file}")
        if not args.dry_run:
            missing = missing_runtime_dependencies(not no_onnx_simplify)
            if missing:
                raise RuntimeError(
                    "Faltan dependencias del pipeline: "
                    + ", ".join(missing)
                    + ". Ejecute: python -m pip install -r requirements.txt "
                    "--extra-index-url https://download.pytorch.org/whl/cu113"
                )

        connection_string = required_env("AZURE_STORAGE_CONNECTION_STRING")
        container = required_env("AZURE_STORAGE_CONTAINER")
        client = create_container_client(connection_string, container)
        existing = [
            blob.name for blob in client.list_blobs(name_starts_with=weights_prefix + "/")
        ]
        latest_version = find_latest_version(existing, weights_prefix)
        expected_target = next_patch_version(latest_version)
        target_version = (
            format_version(version_number(args.version))
            if args.version
            else expected_target
        )
        if args.version and target_version != expected_target:
            raise ValueError(
                f"La siguiente versión de {project} debe ser {expected_target}; "
                f"se recibió {target_version}"
            )
        if latest_version is None and not local_base and not args.allow_no_base:
            raise ValueError(
                f"No hay un modelo base en {container}/{weights_prefix}/; "
                "use --allow-no-base solo para el primer modelo"
            )

        base_version = "none" if local_base else (latest_version or "none")
        base_checkpoint = local_base
        if latest_version and not local_base:
            base_checkpoint = artifacts_project / latest_version / "best_ckpt.pth"

        base_description = (
            f"archivo local: {local_base}" if local_base else
            f"{container}/{weights_prefix}/{latest_version}/best_ckpt.pth" if latest_version else
            "ninguno (inicialización aleatoria)"
        )

        report_dir = output_project / "reports" / target_version / uuid4().hex
        base_metrics = base_checkpoint.with_name("metrics.json") if base_checkpoint else None
        child_environment = os.environ.copy()
        child_environment["YOLOX_REPORT_DIR"] = str(report_dir)
        child_environment["YOLOX_MODEL_VERSION"] = target_version
        child_environment["YOLOX_BASE_VERSION"] = base_version
        child_environment["YOLOX_BASE_SOURCE"] = base_description
        child_environment["YOLOX_DATASET_VERSION"] = dataset_folder
        child_environment.pop("YOLOX_BASE_METRICS", None)
        child_environment["YOLOX_DATA_DIR"] = str(data_dir)
        add_repository_to_pythonpath(child_environment)
        env_args = ["--env-file", str(loaded)] if loaded else []

        clean_command = [sys.executable, "tools/clean_datasets.py", *env_args, "--yes"]
        ingest_command = [
            sys.executable,
            "tools/ingest_blob_storage.py",
            *env_args,
            "--base-prefix",
            blob_base_prefix,
            "--folder",
            dataset_folder,
        ]
        train_command = [
            sys.executable,
            "tools/train.py",
            "--exp_file",
            str(exp_file),
            "--experiment-name",
            project,
            "--devices",
            str(devices),
            "--batch-size",
            str(batch_size),
        ]
        if base_checkpoint:
            train_command.extend(["--ckpt", str(base_checkpoint)])
        if fp16:
            train_command.append("--fp16")
        if extra_train_args:
            train_command.extend(extra_train_args)
        train_command.extend(["output_dir", str(output_dir)])

        publish_command = [
            sys.executable,
            "tools/publish_weights.py",
            "--report-dir",
            str(report_dir),
            *env_args,
            "--ckpt",
            str(checkpoint),
            "--exp-file",
            str(exp_file),
            "--weights-prefix",
            weights_prefix,
            "--version",
            target_version,
            "--project",
            project,
            "--base-version",
            base_version,
            "--dataset-folder",
            dataset_folder,
            "--devices",
            str(devices),
            "--batch-size",
            str(batch_size),
        ]
        if fp16:
            publish_command.append("--fp16")
        if env_bool("PIPELINE_ONNX_DYNAMIC", False):
            publish_command.append("--dynamic")
        if no_onnx_simplify:
            publish_command.append("--no-onnxsim")

        print("Pipeline configurado:")
        print(f"  Prefijo: {prefix}")
        print(f"  Proyecto: {project}")
        print(f"  Modelo base: {base_description}")
        print(f"  Última versión en {weights_prefix}/: {latest_version or 'ninguna'}")
        print(f"  Nueva versión: {target_version}")
        print(f"  Dataset: {container}/{blob_base_prefix}/{dataset_folder}/")
        print(f"  Dataset local: {data_dir}")
        print(f"  Publicación: {container}/{weights_prefix}/{target_version}/")

        print("\n=== 1/6 Preparar modelo base seleccionado ===", flush=True)
        if local_base:
            print(f"Checkpoint local: {local_base}")
        elif base_checkpoint:
            print(f"{weights_prefix}/{latest_version}/best_ckpt.pth -> {base_checkpoint}")
            if not args.dry_run:
                download_base_checkpoint(client, weights_prefix, latest_version, base_checkpoint)
                metrics_blob = combine_prefix(combine_prefix(weights_prefix, latest_version), "metrics.json")
                if metrics_blob in existing:
                    with base_metrics.open("wb") as stream:
                        client.get_blob_client(metrics_blob).download_blob().readinto(stream)
                    child_environment["YOLOX_BASE_METRICS"] = str(base_metrics)
        else:
            print("Primer entrenamiento: no existe un modelo base.")

        if not args.skip_clean:
            run_stage("2/6 Limpiar datasets", clean_command, args.dry_run, child_environment)
        else:
            print("\n=== 2/6 Limpieza inicial omitida ===", flush=True)
        run_stage("3/6 Descargar imágenes", ingest_command, args.dry_run, child_environment)

        previous_mtime = checkpoint.stat().st_mtime_ns if checkpoint.exists() else None
        training_started = time.time_ns()
        run_stage("4/6 Entrenar desde el modelo base", train_command, args.dry_run, child_environment)
        if not args.dry_run:
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise RuntimeError(f"No se generó el checkpoint: {checkpoint}")
            current_mtime = checkpoint.stat().st_mtime_ns
            if previous_mtime is not None and current_mtime == previous_mtime:
                raise RuntimeError("El entrenamiento no actualizó best_ckpt.pth")
            if current_mtime < training_started:
                raise RuntimeError("best_ckpt.pth es anterior a esta ejecución")

        print("\n=== Verificar ficha t?cnica del entrenamiento ===", flush=True)
        print(f"Ficha y m?tricas: {report_dir}")
        if not args.dry_run:
            for name in ("metrics.json", "model_report.html"):
                if not (report_dir / name).is_file():
                    raise RuntimeError(f"No se gener? {report_dir / name}")

        run_stage("5/6 Exportar y publicar", publish_command, args.dry_run, child_environment)

        print("\n=== 6/6 Limpiar artefactos locales ===", flush=True)
        if args.dry_run:
            print("Simulación: se conservaron imágenes y pesos locales.")
        else:
            if not args.skip_clean:
                subprocess.run(clean_command, cwd=REPOSITORY_ROOT, env=child_environment, check=True)
            if not args.keep_local_weights:
                deleted = clean_local_weights(output_project, artifacts_project)
                print(f"Pesos locales eliminados: {deleted}")
            else:
                print("Pesos locales conservados por --keep-local-weights.")

        print("\nPipeline terminado correctamente.")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: una etapa falló con código {exc.returncode}; pipeline detenido", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
