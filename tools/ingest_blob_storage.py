#!/usr/bin/env python3
"""Descarga una carpeta de Azure Blob Storage dentro de datasets/."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def normalize_blob_folder(value: str) -> str:
    """Normaliza una carpeta virtual de Blob Storage y rechaza rutas inseguras."""
    normalized = value.strip().replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError("--folder debe ser una ruta relativa válida dentro del contenedor")
    return path.as_posix()


def combine_prefix(base_prefix: str, folder: str) -> str:
    parts = [part.strip("/") for part in (base_prefix, folder) if part.strip("/")]
    return "/".join(parts)


def safe_destination(root: Path, relative_blob: str) -> Path:
    """Resuelve el destino asegurando que permanezca debajo de root."""
    relative = PurePosixPath(relative_blob)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Ruta de blob insegura: {relative_blob}")
    resolved_root = root.resolve()
    destination = resolved_root.joinpath(*relative.parts).resolve()
    if resolved_root not in destination.parents:
        raise ValueError(f"El blob sale de la carpeta destino: {relative_blob}")
    return destination


def load_environment(env_file: str | None) -> Path | None:
    """Carga .env sin reemplazar variables ya definidas en el sistema."""
    configured = env_file or os.getenv("YOLOX_ENV_FILE")
    path = Path(configured).expanduser() if configured else REPOSITORY_ROOT / ".env"
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.exists():
        load_dotenv(path, override=False)
        return path
    if configured:
        raise ValueError(f"No existe el archivo de variables: {path}")
    return None


def create_container_client(connection_string: str, container: str):
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:
        raise RuntimeError(
            "Falta azure-storage-blob. Ejecute: python -m pip install -r requirements.txt"
        ) from exc
    service = BlobServiceClient.from_connection_string(connection_string)
    return service.get_container_client(container)


def download_one(container_client, blob_name: str, destination: Path, overwrite: bool) -> str:
    blob_client = container_client.get_blob_client(blob_name)
    properties = blob_client.get_blob_properties()
    expected_size = int(properties.size or 0)
    if destination.exists() and not overwrite and destination.stat().st_size == expected_size:
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with temporary.open("wb") as output:
            blob_client.download_blob(max_concurrency=1).readinto(output)
        if temporary.stat().st_size != expected_size:
            raise IOError(
                f"Tamaño incorrecto para {blob_name}: "
                f"{temporary.stat().st_size} != {expected_size}"
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "downloaded"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Descarga una carpeta/prefijo de Azure Blob Storage en datasets/."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Carpeta virtual dentro del contenedor, por ejemplo COCO o proyectos/lote-01.",
    )
    parser.add_argument(
        "--destination",
        default=None,
        help="Raíz local. Por defecto usa AZURE_INGEST_DESTINATION o datasets.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Archivo .env. Por defecto usa .env en la raíz del repositorio.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Descargas simultáneas.")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe archivos existentes.")
    parser.add_argument("--dry-run", action="store_true", help="Lista sin descargar.")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        if args.workers < 1:
            raise ValueError("--workers debe ser mayor que cero")
        loaded = load_environment(args.env_file)
        if loaded:
            print(f"Configuración cargada desde: {loaded}")

        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container = os.getenv("AZURE_STORAGE_CONTAINER")
        if not connection_string:
            raise ValueError("Falta AZURE_STORAGE_CONNECTION_STRING en el entorno o .env")
        if not container:
            raise ValueError("Falta AZURE_STORAGE_CONTAINER en el entorno o .env")

        folder = normalize_blob_folder(args.folder)
        base_prefix = os.getenv("AZURE_BLOB_BASE_PREFIX", "")
        azure_prefix = combine_prefix(base_prefix, folder)
        listing_prefix = azure_prefix + "/"

        configured_destination = args.destination or os.getenv(
            "AZURE_INGEST_DESTINATION", "datasets"
        )
        destination_root = Path(configured_destination).expanduser()
        if not destination_root.is_absolute():
            destination_root = REPOSITORY_ROOT / destination_root
        local_dataset = safe_destination(destination_root, folder)

        client = create_container_client(connection_string, container)
        blobs = [
            blob
            for blob in client.list_blobs(name_starts_with=listing_prefix)
            if blob.name and not blob.name.endswith("/")
        ]
        if not blobs:
            raise ValueError(
                f"No se encontraron archivos en {container}/{listing_prefix}"
            )

        print(f"Origen: contenedor={container}, carpeta={azure_prefix}")
        print(f"Destino: {local_dataset}")
        print(f"Archivos encontrados: {len(blobs)}")

        jobs = []
        for blob in blobs:
            relative = blob.name[len(listing_prefix):]
            jobs.append((blob.name, safe_destination(local_dataset, relative)))

        if args.dry_run:
            for blob_name, destination in jobs:
                print(f"[DRY RUN] {blob_name} -> {destination}")
            return 0

        counts = {"downloaded": 0, "skipped": 0}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_one, client, blob_name, destination, args.overwrite): blob_name
                for blob_name, destination in jobs
            }
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                counts[result] += 1
                if completed % 100 == 0 or completed == len(futures):
                    print(f"Procesados: {completed}/{len(futures)}")

        print(
            f"Ingesta terminada: descargados={counts['downloaded']}, "
            f"omitidos={counts['skipped']}"
        )
        print(f"Configure YOLOX_DATA_DIR={local_dataset}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
