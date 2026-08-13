#!/usr/bin/env python3
"""Pipeline segmentado: modelo base, datos, entrenamiento y publicación."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.ingest_blob_storage import (  # noqa: E402
    combine_prefix,
    create_container_client,
    load_environment,
)
from tools.publish_weights import (  # noqa: E402
    VERSION_PATTERN,
    format_version,
    next_patch_version,
    normalize_prefix,
    version_number,
)


PROJECT_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


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


def project_key(project: str) -> str:
    normalized = project.strip().lower()
    if not PROJECT_PATTERN.fullmatch(normalized):
        raise ValueError("--project solo admite letras minúsculas, números, _ y -")
    return normalized.upper().replace("-", "_")


def project_env(key: str, suffix: str) -> str:
    return required_env(f"{key}_{suffix}")


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def display_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


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
        "--project",
        required=True,
        help="Perfil del proyecto, por ejemplo vet_yolox o lis_yolox.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "Versión SemVer opcional. Por defecto aumenta automáticamente "
            "el parche de la última versión."
        ),
    )
    parser.add_argument("--env-file", default=None, help="Archivo .env alternativo.")
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
        help="Permite iniciar 1.0.0 sin un modelo previamente publicado.",
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

        project = args.project.strip().lower()
        key = project_key(project)
        dataset_folder = project_env(key, "DATASET_FOLDER")
        blob_base_prefix = project_env(key, "BLOB_BASE_PREFIX")
        weights_prefix = normalize_prefix(project_env(key, "WEIGHTS_PREFIX"))
        exp_file = resolve_repo_path(project_env(key, "EXP_FILE"))
        output_dir = resolve_repo_path(os.getenv("PIPELINE_OUTPUT_DIR", "YOLOX_outputs"))
        output_project = output_dir / project
        checkpoint = output_project / "best_ckpt.pth"
        artifacts_project = resolve_repo_path(
            os.getenv("PIPELINE_ARTIFACTS_DIR", ".pipeline_artifacts")
        ) / project
        devices = int(os.getenv("PIPELINE_DEVICES", "1"))
        batch_size = int(os.getenv("PIPELINE_BATCH_SIZE", "8"))
        fp16 = env_bool("PIPELINE_FP16", True)
        if devices < 1 or batch_size < 1:
            raise ValueError("PIPELINE_DEVICES y PIPELINE_BATCH_SIZE deben ser mayores que cero")
        if not exp_file.is_file():
            raise ValueError(f"No existe el experimento configurado: {exp_file}")

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
        if latest_version is None and not args.allow_no_base:
            raise ValueError(
                f"No hay un modelo base en {container}/{weights_prefix}/; "
                "use --allow-no-base solo para publicar 1.0.0"
            )

        base_checkpoint = None
        if latest_version:
            base_checkpoint = artifacts_project / latest_version / "best_ckpt.pth"

        profile_overrides = {
            "YOLOX_DATA_DIR": project_env(key, "DATA_DIR"),
            "YOLOX_TRAIN_IMAGES": project_env(key, "TRAIN_IMAGES"),
            "YOLOX_VAL_IMAGES": project_env(key, "VAL_IMAGES"),
            "YOLOX_TEST_IMAGES": project_env(key, "TEST_IMAGES"),
            "YOLOX_ANNOTATIONS_DIR": project_env(key, "ANNOTATIONS_DIR"),
            "YOLOX_TRAIN_ANN": project_env(key, "TRAIN_ANN"),
            "YOLOX_VAL_ANN": project_env(key, "VAL_ANN"),
            "YOLOX_TEST_ANN": project_env(key, "TEST_ANN"),
        }
        child_environment = os.environ.copy()
        child_environment.update(profile_overrides)
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
        extra_train_args = os.getenv("PIPELINE_TRAIN_ARGS", "").strip()
        if extra_train_args:
            train_command.extend(shlex.split(extra_train_args))
        train_command.extend(["output_dir", str(output_dir)])

        publish_command = [
            sys.executable,
            "tools/publish_weights.py",
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
            latest_version or "none",
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
        if env_bool("PIPELINE_ONNX_NO_SIMPLIFY", False):
            publish_command.append("--no-onnxsim")

        print("Pipeline configurado:")
        print(f"  Proyecto: {project}")
        print(f"  Modelo base: {latest_version or 'ninguno'}")
        print(f"  Nueva versión: {target_version}")
        print(f"  Dataset: {container}/{blob_base_prefix}/{dataset_folder}/")
        print(f"  Publicación: {container}/{weights_prefix}/{target_version}/")

        print("\n=== 1/6 Obtener último modelo base ===", flush=True)
        if base_checkpoint:
            print(f"{weights_prefix}/{latest_version}/best_ckpt.pth -> {base_checkpoint}")
            if not args.dry_run:
                download_base_checkpoint(client, weights_prefix, latest_version, base_checkpoint)
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
