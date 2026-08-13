#!/usr/bin/env python3
"""Exporta el mejor checkpoint a ONNX y publica ambos formatos en Azure."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

try:
    from tools.ingest_blob_storage import (
        REPOSITORY_ROOT,
        combine_prefix,
        create_container_client,
        load_environment,
    )
except ModuleNotFoundError:  # Ejecución directa: python tools/publish_weights.py
    from ingest_blob_storage import (
        REPOSITORY_ROOT,
        combine_prefix,
        create_container_client,
        load_environment,
    )


VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
PROJECT_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


def normalize_prefix(value: str) -> str:
    normalized = value.strip().replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError("AZURE_WEIGHTS_PREFIX debe ser una ruta relativa válida")
    return path.as_posix()


def version_number(value: str) -> tuple[int, int, int]:
    match = VERSION_PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError("La versión debe tener formato SemVer, por ejemplo 1.0.1")
    return tuple(int(part) for part in match.groups())


def format_version(version: tuple[int, int, int]) -> str:
    if any(part < 0 for part in version):
        raise ValueError("Los componentes de la versión no pueden ser negativos")
    return ".".join(str(part) for part in version)


def next_patch_version(version: str | None) -> str:
    if version is None:
        return "1.0.0"
    major, minor, patch = version_number(version)
    return format_version((major, minor, patch + 1))


def find_next_version(blob_names: list[str], prefix: str) -> str:
    base = prefix.rstrip("/") + "/"
    versions = []
    for blob_name in blob_names:
        if not blob_name.startswith(base):
            continue
        relative = blob_name[len(base):]
        first_part = relative.split("/", 1)[0]
        match = VERSION_PATTERN.fullmatch(first_part)
        if match:
            versions.append(tuple(int(part) for part in match.groups()))
    latest = format_version(max(versions)) if versions else None
    return next_patch_version(latest)


def upload_file(container_client, source: Path, blob_name: str) -> None:
    with source.open("rb") as stream:
        container_client.upload_blob(name=blob_name, data=stream, overwrite=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, object]:
    try:
        import torch

        state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            return {}
        return {
            key: state[key]
            for key in ("start_epoch", "best_ap", "curr_ap")
            if key in state and isinstance(state[key], (int, float, str, type(None)))
        }
    except Exception:
        return {}


def export_onnx(
    checkpoint: Path,
    output: Path,
    exp_file: Path,
    opset: int,
    dynamic: bool,
    no_onnxsim: bool,
) -> None:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "tools" / "export_onnx.py"),
        "--exp_file",
        str(exp_file),
        "--ckpt",
        str(checkpoint),
        "--output-name",
        str(output),
        "--opset",
        str(opset),
    ]
    if dynamic:
        command.append("--dynamic")
    if no_onnxsim:
        command.append("--no-onnxsim")
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publica el checkpoint, el ONNX del proyecto y README en Azure."
    )
    parser.add_argument("--ckpt", required=True, help="Ruta del mejor checkpoint .pth")
    parser.add_argument(
        "--exp-file",
        default="exps/cassette/cassette_yolox.py",
        help="Archivo de experimento usado para exportar ONNX.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Versión SemVer explícita, por ejemplo 1.0.1. Por defecto usa la siguiente.",
    )
    parser.add_argument("--env-file", default=None, help="Archivo .env alternativo.")
    parser.add_argument(
        "--weights-prefix",
        default=None,
        help="Prefijo remoto. Por defecto usa AZURE_WEIGHTS_PREFIX o weights.",
    )
    parser.add_argument("--project", required=True, help="Nombre del proyecto.")
    parser.add_argument("--base-version", default="none", help="Versión usada como base.")
    parser.add_argument("--dataset-folder", default="unspecified", help="Lote entrenado.")
    parser.add_argument("--devices", type=int, default=1, help="Cantidad de GPU usadas.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch del entrenamiento.")
    parser.add_argument("--fp16", action="store_true", help="Indica entrenamiento FP16.")
    parser.add_argument("--opset", type=int, default=11, help="Versión ONNX opset.")
    parser.add_argument("--dynamic", action="store_true", help="Batch dinámico en ONNX.")
    parser.add_argument(
        "--no-onnxsim",
        action="store_true",
        help="No simplifica el archivo ONNX exportado.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcula la versión y las rutas sin exportar ni subir archivos.",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        loaded = load_environment(args.env_file)
        if loaded:
            print(f"Configuración cargada desde: {loaded}")

        checkpoint = Path(args.ckpt).expanduser().resolve()
        exp_file = Path(args.exp_file).expanduser()
        if not exp_file.is_absolute():
            exp_file = (REPOSITORY_ROOT / exp_file).resolve()
        if not checkpoint.is_file():
            raise ValueError(f"No existe el checkpoint: {checkpoint}")
        if checkpoint.suffix.lower() not in {".pth", ".pt"}:
            raise ValueError("El checkpoint debe tener extensión .pth o .pt")
        if not exp_file.is_file():
            raise ValueError(f"No existe el experimento: {exp_file}")
        if args.opset < 1:
            raise ValueError("--opset debe ser mayor que cero")

        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container = os.getenv("AZURE_STORAGE_CONTAINER")
        if not connection_string:
            raise ValueError("Falta AZURE_STORAGE_CONNECTION_STRING en el entorno o .env")
        if not container:
            raise ValueError("Falta AZURE_STORAGE_CONTAINER en el entorno o .env")

        prefix = normalize_prefix(
            args.weights_prefix or os.getenv("AZURE_WEIGHTS_PREFIX", "weights")
        )
        client = create_container_client(connection_string, container)
        existing = [blob.name for blob in client.list_blobs(name_starts_with=prefix + "/")]

        if args.version:
            version = format_version(version_number(args.version))
        else:
            version = find_next_version(existing, prefix)

        project = args.project.strip().lower()
        if not PROJECT_PATTERN.fullmatch(project):
            raise ValueError("--project no es válido para nombrar el archivo ONNX")

        version_prefix = combine_prefix(prefix, version)
        original_blob = combine_prefix(version_prefix, "best_ckpt.pth")
        onnx_name = f"{project}.onnx"
        onnx_blob = combine_prefix(version_prefix, onnx_name)
        readme_blob = combine_prefix(version_prefix, "README.md")
        occupied = {original_blob, onnx_blob, readme_blob}.intersection(existing)
        if occupied:
            raise ValueError(
                f"La versión {version} ya contiene pesos; seleccione otra versión"
            )

        print(f"Contenedor: {container}")
        print(f"Versión: {version}")
        print(f"Checkpoint: {checkpoint} -> {original_blob}")
        print(f"ONNX: {onnx_blob}")
        print(f"Informe: {readme_blob}")
        if args.dry_run:
            print("Simulación terminada: no se exportó ni subió ningún archivo.")
            return 0

        with tempfile.TemporaryDirectory(prefix="yolox-onnx-") as directory:
            onnx_file = Path(directory) / onnx_name
            print("Exportando el mejor checkpoint a ONNX...")
            export_onnx(
                checkpoint,
                onnx_file,
                exp_file,
                args.opset,
                args.dynamic,
                args.no_onnxsim,
            )
            if not onnx_file.is_file() or onnx_file.stat().st_size == 0:
                raise RuntimeError("La exportación no generó un archivo ONNX válido")

            from datetime import datetime, timezone

            readme_file = Path(directory) / "README.md"
            metadata = checkpoint_metadata(checkpoint)
            metadata_lines = [
                f"- Época guardada: `{metadata.get('start_epoch', 'no disponible')}`",
                f"- Mejor AP: `{metadata.get('best_ap', 'no disponible')}`",
                f"- AP actual: `{metadata.get('curr_ap', 'no disponible')}`",
            ]
            readme_file.write_text(
                "\n".join(
                    [
                        f"# Modelo {args.project} {version}",
                        "",
                        f"- Proyecto: `{args.project}`",
                        f"- Versión: `{version}`",
                        f"- Modelo base: `{args.base_version}`",
                        f"- Dataset: `{args.dataset_folder}`",
                        f"- Experimento: `{exp_file}`",
                        f"- GPU utilizadas: `{args.devices}`",
                        f"- Batch size: `{args.batch_size}`",
                        f"- FP16: `{args.fp16}`",
                        f"- Publicado UTC: `{datetime.now(timezone.utc).isoformat()}`",
                        f"- Tamaño PTH: `{checkpoint.stat().st_size}` bytes",
                        f"- Tamaño ONNX: `{onnx_file.stat().st_size}` bytes",
                        f"- SHA-256 PTH: `{sha256_file(checkpoint)}`",
                        f"- SHA-256 ONNX: `{sha256_file(onnx_file)}`",
                        *metadata_lines,
                        "",
                        "## Archivos",
                        "",
                        "- `best_ckpt.pth`: checkpoint original de YOLOX.",
                        f"- `{onnx_name}`: modelo exportado para inferencia.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            print("Subiendo pesos e informe a Azure...")
            uploaded = []
            try:
                upload_file(client, checkpoint, original_blob)
                uploaded.append(original_blob)
                upload_file(client, onnx_file, onnx_blob)
                uploaded.append(onnx_blob)
                upload_file(client, readme_file, readme_blob)
                uploaded.append(readme_blob)
            except Exception:
                for blob_name in uploaded:
                    client.delete_blob(blob_name)
                raise

        print(f"Publicación terminada: {container}/{version_prefix}/")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: falló la exportación ONNX (código {exc.returncode})", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
