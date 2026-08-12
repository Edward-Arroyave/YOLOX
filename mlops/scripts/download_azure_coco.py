#!/usr/bin/env python3
"""Download a date-bounded COCO dataset from Azure Blob Storage.

Image blobs are selected using their Azure ``last_modified`` timestamp in the
half-open UTC interval [start, end). Annotation JSON files are always downloaded
and then filtered so they reference only the selected local images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SelectedBlob:
    name: str
    local_path: str
    split: str
    etag: str
    size: int
    last_modified: str


def env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def parse_utc(value: str, name: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone, preferably Z for UTC")
    return parsed.astimezone(timezone.utc)


def join_blob_path(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def safe_local_path(root: Path, blob_name: str, source_prefix: str) -> Path:
    prefix = source_prefix.strip("/")
    relative = blob_name[len(prefix):].lstrip("/") if prefix else blob_name
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError(f"Unsafe blob path: {blob_name}")
    destination = root.joinpath(*posix.parts).resolve()
    resolved_root = root.resolve()
    if destination != resolved_root and resolved_root not in destination.parents:
        raise ValueError(f"Blob path escapes destination: {blob_name}")
    return destination


def build_container_client(account_url: str, container: str):
    try:
        from azure.identity import EnvironmentCredential
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:
        raise RuntimeError(
            "Azure SDK is missing. Run: python -m pip install -r requirements-mlops.txt"
        ) from exc

    credential = EnvironmentCredential()
    service = BlobServiceClient(account_url=account_url, credential=credential)
    return service.get_container_client(container)


def list_selected_images(
    container_client: Any,
    destination: Path,
    source_prefix: str,
    split_dirs: dict[str, str],
    start: datetime,
    end: datetime,
) -> list[SelectedBlob]:
    selected: list[SelectedBlob] = []
    for split, image_dir in split_dirs.items():
        blob_prefix = join_blob_path(source_prefix, image_dir) + "/"
        for blob in container_client.list_blobs(name_starts_with=blob_prefix):
            suffix = PurePosixPath(blob.name).suffix.lower()
            modified = blob.last_modified.astimezone(timezone.utc)
            if suffix not in IMAGE_EXTENSIONS or not (start <= modified < end):
                continue
            local = safe_local_path(destination, blob.name, source_prefix)
            selected.append(
                SelectedBlob(
                    name=blob.name,
                    local_path=str(local),
                    split=split,
                    etag=str(blob.etag or "").strip('"'),
                    size=int(blob.size or 0),
                    last_modified=modified.isoformat().replace("+00:00", "Z"),
                )
            )
    return selected


def download_blob(container_client: Any, blob_name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with temporary.open("wb") as output:
            stream = container_client.download_blob(blob_name)
            stream.readinto(output)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def filter_coco(
    annotation_path: Path,
    available: set[str],
    output_path: Path,
) -> dict[str, int]:
    with annotation_path.open("r", encoding="utf-8") as source:
        coco = json.load(source)

    available_names = {PurePosixPath(name).name for name in available}

    def image_exists(file_name: str) -> bool:
        normalized = PurePosixPath(file_name.replace("\\", "/")).as_posix().lstrip("./")
        return normalized in available or PurePosixPath(normalized).name in available_names

    images = [image for image in coco.get("images", []) if image_exists(str(image["file_name"]))]
    image_ids = {image["id"] for image in images}
    annotations = [ann for ann in coco.get("annotations", []) if ann.get("image_id") in image_ids]
    coco["images"] = images
    coco["annotations"] = annotations

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(coco, target, ensure_ascii=False, indent=2)
        target.write("\n")
    temporary.replace(output_path)
    return {"images": len(images), "annotations": len(annotations)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--start", default=env("AZURE_DATA_START"))
    result.add_argument("--end", default=env("AZURE_DATA_END"))
    result.add_argument("--destination", default=env("AZURE_DATASET_DIR"))
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        account_url = str(env("AZURE_STORAGE_ACCOUNT_URL", required=True))
        container = str(env("AZURE_STORAGE_CONTAINER", required=True))
        prefix = str(env("AZURE_STORAGE_PREFIX", ""))
        if not args.start or not args.end or not args.destination:
            raise ValueError("AZURE_DATA_START, AZURE_DATA_END and AZURE_DATASET_DIR are required")
        start = parse_utc(args.start, "AZURE_DATA_START")
        end = parse_utc(args.end, "AZURE_DATA_END")
        if start >= end:
            raise ValueError("AZURE_DATA_START must be earlier than AZURE_DATA_END")
        destination = Path(args.destination).expanduser().resolve()

        split_dirs: dict[str, str] = {
            "train": str(env("AZURE_TRAIN_IMAGE_DIR", "train")),
            "val": str(env("AZURE_VAL_IMAGE_DIR", "val")),
        }
        annotation_blobs: dict[str, str] = {
            "train": str(env("AZURE_TRAIN_ANNOTATION_BLOB", "annotations/train.json")),
            "val": str(env("AZURE_VAL_ANNOTATION_BLOB", "annotations/val.json")),
        }
        test_dir = env("AZURE_TEST_IMAGE_DIR")
        test_ann = env("AZURE_TEST_ANNOTATION_BLOB")
        if bool(test_dir) != bool(test_ann):
            raise ValueError(
                "AZURE_TEST_IMAGE_DIR and AZURE_TEST_ANNOTATION_BLOB must be set together"
            )
        if test_dir and test_ann:
            split_dirs["test"] = test_dir
            annotation_blobs["test"] = test_ann

        client = build_container_client(account_url, container)
        selected = list_selected_images(client, destination, prefix, split_dirs, start, end)
        print(f"Selected {len(selected)} image blobs in [{start.isoformat()}, {end.isoformat()})")
        for split in split_dirs:
            print(f"  {split}: {sum(item.split == split for item in selected)}")
        if not selected:
            raise ValueError("No images matched the requested date interval")
        for required_split in ("train", "val"):
            if not any(item.split == required_split for item in selected):
                raise ValueError(
                    f"No {required_split} images matched the requested date interval"
                )
        if args.dry_run:
            return 0

        destination.mkdir(parents=True, exist_ok=True)
        for index, blob in enumerate(selected, 1):
            download_blob(client, blob.name, Path(blob.local_path))
            if index % 100 == 0 or index == len(selected):
                print(f"Downloaded {index}/{len(selected)} images")

        filtered_counts: dict[str, dict[str, int]] = {}
        annotation_checksums: dict[str, str] = {}
        for split, relative_blob in annotation_blobs.items():
            source_blob = join_blob_path(prefix, relative_blob)
            annotation_path = safe_local_path(destination, source_blob, prefix)
            download_blob(client, source_blob, annotation_path)
            image_root = (destination / split_dirs[split]).resolve()
            selected_files = {
                Path(item.local_path).resolve().relative_to(image_root).as_posix()
                for item in selected
                if item.split == split
            }
            filtered_counts[split] = filter_coco(annotation_path, selected_files, annotation_path)
            annotation_checksums[split] = sha256(annotation_path)

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": {
                "account_url": account_url,
                "container": container,
                "prefix": prefix,
                "start_inclusive": start.isoformat().replace("+00:00", "Z"),
                "end_exclusive": end.isoformat().replace("+00:00", "Z"),
            },
            "splits": filtered_counts,
            "annotation_sha256": annotation_checksums,
            "objects": [asdict(item) for item in selected],
        }
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Dataset ready: {destination}")
        print(f"Manifest: {manifest_path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
