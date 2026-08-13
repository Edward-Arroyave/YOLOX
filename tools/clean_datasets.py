#!/usr/bin/env python3
"""Limpia el contenido del directorio local de datasets de forma segura."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    from tools.ingest_blob_storage import REPOSITORY_ROOT, load_environment
except ModuleNotFoundError:  # Ejecución directa: python tools/clean_datasets.py
    from ingest_blob_storage import REPOSITORY_ROOT, load_environment


def resolve_cleanup_target(configured_destination: str) -> Path:
    """Resuelve el destino y rechaza directorios demasiado amplios."""
    target = Path(configured_destination).expanduser()
    if not target.is_absolute():
        target = REPOSITORY_ROOT / target
    target = target.resolve()

    forbidden = {
        Path(target.anchor).resolve(),
        Path.home().resolve(),
        REPOSITORY_ROOT.resolve(),
    }
    if target in forbidden:
        raise ValueError(f"Ruta de limpieza insegura: {target}")
    return target


def inventory(target: Path) -> tuple[int, int, int]:
    """Devuelve cantidad de archivos, directorios y bytes del destino."""
    if not target.exists():
        return 0, 0, 0
    if not target.is_dir():
        raise ValueError(f"El destino no es un directorio: {target}")

    files = directories = total_bytes = 0
    for item in target.rglob("*"):
        if item.is_symlink() or item.is_file():
            files += 1
            try:
                total_bytes += item.stat().st_size
            except OSError:
                pass
        elif item.is_dir():
            directories += 1
    return files, directories, total_bytes


def clean_contents(target: Path) -> None:
    """Elimina los elementos hijos, sin eliminar el directorio raíz."""
    target.mkdir(parents=True, exist_ok=True)
    for item in target.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Elimina todo el contenido del directorio local de datasets."
    )
    parser.add_argument(
        "--destination",
        default=None,
        help="Directorio a limpiar. Usa AZURE_INGEST_DESTINATION o datasets.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Archivo .env. Por defecto usa .env en la raíz del repositorio.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra qué se limpiaría sin eliminar archivos.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirma la eliminación. Es obligatorio fuera de --dry-run.",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        loaded = load_environment(args.env_file)
        if loaded:
            print(f"Configuración cargada desde: {loaded}")

        configured = args.destination or os.getenv(
            "AZURE_INGEST_DESTINATION", "datasets"
        )
        target = resolve_cleanup_target(configured)
        files, directories, total_bytes = inventory(target)

        print(f"Destino: {target}")
        print(
            f"Contenido: archivos={files}, directorios={directories}, "
            f"bytes={total_bytes}"
        )

        if args.dry_run:
            print("Simulación terminada: no se eliminó ningún archivo.")
            return 0
        if not args.yes:
            raise ValueError("La limpieza requiere --yes")

        clean_contents(target)
        print(f"Limpieza terminada: se conservó el directorio {target}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
