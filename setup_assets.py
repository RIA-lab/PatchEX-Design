from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


ZENODO_RECORDS = {
    "ped-eval": 19992035,
    "pipeline": 19992598,
}

REQUIRED_DATA_DESIGN_ITEMS = {
    "opt.csv",
    "ph.csv",
    "pdb",
    "ec_pools",
    "functional_site_annotations",
}
REQUIRED_PREDICTOR_WEIGHT_ITEMS = {
    "ephod",
    "patchex_weight",
    "patchet_pretrain_weight",
}
REQUIRED_PIPELINE_WEIGHT_ITEMS = {"opt", "ph"}


class AssetSetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class AssetTarget:
    bundle: str
    name: str
    destination: Path
    kind: str
    validator: Callable[[Path], bool]
    finder: Callable[[Path], Path | None]

    def is_installed(self) -> bool:
        return self.validator(self.destination)


def _directory_with_children_validator(required_children: set[str]) -> Callable[[Path], bool]:
    def _validator(path: Path) -> bool:
        if not path.is_dir():
            return False
        child_names = {child.name for child in path.iterdir()}
        return required_children.issubset(child_names)

    return _validator


def _existing_directory(path: Path) -> bool:
    return path.is_dir()


def _existing_file(path: Path) -> bool:
    return path.is_file()


def _find_named_path(stage_root: Path, name: str, *, kind: str = "dir", exclude_parts: set[str] | None = None) -> Path | None:
    exclude_parts = exclude_parts or set()
    matches: list[Path] = []
    for path in stage_root.rglob(name):
        if exclude_parts.intersection(path.parts):
            continue
        if kind == "dir" and path.is_dir():
            matches.append(path)
        elif kind == "file" and path.is_file():
            matches.append(path)
    if not matches:
        return None
    matches.sort(key=lambda match_path: (len(match_path.parts), str(match_path)))
    return matches[0]


def _find_parent_with_children(stage_root: Path, required_children: set[str], *, exclude_parts: set[str] | None = None) -> Path | None:
    exclude_parts = exclude_parts or set()
    candidates: list[Path] = []
    for path in stage_root.rglob("*"):
        if exclude_parts.intersection(path.parts) or not path.is_dir():
            continue
        child_names = {child.name for child in path.iterdir()}
        if required_children.issubset(child_names):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda match_dir: (len(match_dir.parts), str(match_dir)))
    return candidates[0]


def build_asset_targets(repo_root: Path) -> dict[str, AssetTarget]:
    return {
        "data_design": AssetTarget(
            bundle="ped-eval",
            name="data_design",
            destination=repo_root / "data_design",
            kind="dir",
            validator=_directory_with_children_validator(REQUIRED_DATA_DESIGN_ITEMS),
            finder=lambda stage_root: _find_named_path(stage_root, "data_design", kind="dir")
            or _find_parent_with_children(stage_root, REQUIRED_DATA_DESIGN_ITEMS),
        ),
        "predictor_weights": AssetTarget(
            bundle="ped-eval",
            name="predictor_weights",
            destination=repo_root / "predictor_weights",
            kind="dir",
            validator=_directory_with_children_validator(REQUIRED_PREDICTOR_WEIGHT_ITEMS),
            finder=lambda stage_root: _find_named_path(stage_root, "predictor_weights", kind="dir")
            or _find_parent_with_children(stage_root, REQUIRED_PREDICTOR_WEIGHT_ITEMS),
        ),
        "mapdiff_weight": AssetTarget(
            bundle="pipeline",
            name="mapdiff_weight.pt",
            destination=repo_root / "MapDiff" / "mapdiff_weight.pt",
            kind="file",
            validator=_existing_file,
            finder=lambda stage_root: _find_named_path(stage_root, "mapdiff_weight.pt", kind="file"),
        ),
        "patchex_weight": AssetTarget(
            bundle="pipeline",
            name="patchex_weight",
            destination=repo_root / "patchex_weight",
            kind="dir",
            validator=_directory_with_children_validator(REQUIRED_PIPELINE_WEIGHT_ITEMS),
            finder=lambda stage_root: _find_parent_with_children(
                stage_root,
                REQUIRED_PIPELINE_WEIGHT_ITEMS,
                exclude_parts={"predictor_weights"},
            ),
        ),
        "esm150": AssetTarget(
            bundle="pipeline",
            name="esm150",
            destination=repo_root / "esm150",
            kind="dir",
            validator=_existing_directory,
            finder=lambda stage_root: _find_named_path(stage_root, "esm150", kind="dir"),
        ),
    }


def _download_file(url: str, destination: Path, *, quiet: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"[INFO] Downloading {url} -> {destination}")
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
    except urllib.error.URLError as exc:
        raise AssetSetupError(f"Failed to download {url}: {exc}") from exc
    return destination


def _fetch_zenodo_record_files(record_id: int, download_dir: Path, *, quiet: bool = False) -> list[Path]:
    record_url = f"https://zenodo.org/api/records/{record_id}"
    if not quiet:
        print(f"[INFO] Resolving Zenodo record {record_id}")
    try:
        with urllib.request.urlopen(record_url) as response:
            payload = json.load(response)
    except urllib.error.URLError as exc:
        raise AssetSetupError(f"Failed to resolve Zenodo record {record_id}: {exc}") from exc

    files = payload.get("files") or []
    if not files:
        raise AssetSetupError(f"Zenodo record {record_id} did not return any downloadable files.")

    downloaded_files: list[Path] = []
    for file_entry in files:
        raw_file_name = file_entry.get("key") or file_entry.get("filename") or ""
        if not raw_file_name:
            continue
        file_name = Path(raw_file_name).name
        download_url = (
            file_entry.get("links", {}).get("self")
            or file_entry.get("links", {}).get("download")
            or file_entry.get("links", {}).get("content")
        )
        if not file_name or not download_url:
            continue
        downloaded_files.append(_download_file(download_url, download_dir / file_name, quiet=quiet))

    if not downloaded_files:
        raise AssetSetupError(f"Zenodo record {record_id} did not provide downloadable file links.")
    return downloaded_files


def _stage_downloads(downloaded_files: Iterable[Path], stage_root: Path, *, quiet: bool = False) -> None:
    for file_path in downloaded_files:
        target_dir = stage_root / file_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(file_path):
            if not quiet:
                print(f"[INFO] Extracting zip archive {file_path.name}")
            with zipfile.ZipFile(file_path) as archive:
                archive.extractall(target_dir)
            continue

        if tarfile.is_tarfile(file_path):
            if not quiet:
                print(f"[INFO] Extracting tar archive {file_path.name}")
            with tarfile.open(file_path) as archive:
                archive.extractall(target_dir)
            continue

        shutil.copy2(file_path, target_dir / file_path.name)


def _install_directory(source: Path, destination: Path, *, force: bool = False, quiet: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force and destination.exists():
        shutil.rmtree(destination)
    if not quiet:
        print(f"[INFO] Installing directory {source} -> {destination}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _install_file(source: Path, destination: Path, *, force: bool = False, quiet: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        if not quiet:
            print(f"[INFO] Keeping existing file {destination}")
        return
    if not quiet:
        print(f"[INFO] Installing file {source} -> {destination}")
    shutil.copy2(source, destination)


def _resolve_missing_targets(targets: dict[str, AssetTarget], bundles: Iterable[str], *, force: bool = False) -> list[AssetTarget]:
    requested = set(bundles)
    missing: list[AssetTarget] = []
    for target in targets.values():
        if target.bundle not in requested:
            continue
        if force or not target.is_installed():
            missing.append(target)
    return missing


def ensure_assets(
    bundles: Iterable[str] | None = None,
    *,
    repo_root: str | Path | None = None,
    force: bool = False,
    quiet: bool = False,
) -> None:
    requested_bundles = tuple(dict.fromkeys(bundles or ZENODO_RECORDS.keys()))
    invalid_bundles = [bundle for bundle in requested_bundles if bundle not in ZENODO_RECORDS]
    if invalid_bundles:
        raise AssetSetupError(f"Unsupported asset bundle(s): {', '.join(invalid_bundles)}")

    resolved_repo_root = Path(repo_root or Path(__file__).resolve().parent).resolve()
    targets = build_asset_targets(resolved_repo_root)
    missing_targets = _resolve_missing_targets(targets, requested_bundles, force=force)

    if not missing_targets:
        if not quiet:
            print("[INFO] Requested assets are already installed.")
        return

    record_ids = list(dict.fromkeys(ZENODO_RECORDS[bundle] for bundle in requested_bundles))
    if not quiet:
        missing_names = ", ".join(sorted(target.name for target in missing_targets))
        print(f"[INFO] Preparing missing assets: {missing_names}")

    with tempfile.TemporaryDirectory(prefix="patchex_assets_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        download_dir = temp_dir / "downloads"
        stage_root = temp_dir / "staging"
        download_dir.mkdir(parents=True, exist_ok=True)
        stage_root.mkdir(parents=True, exist_ok=True)

        for record_id in record_ids:
            downloaded_files = _fetch_zenodo_record_files(record_id, download_dir, quiet=quiet)
            _stage_downloads(downloaded_files, stage_root, quiet=quiet)

        for target in missing_targets:
            source = target.finder(stage_root)
            if source is None:
                raise AssetSetupError(
                    f"Downloaded assets for bundle '{target.bundle}' did not contain '{target.name}'."
                )
            if target.kind == "dir":
                _install_directory(source, target.destination, force=force, quiet=quiet)
            else:
                _install_file(source, target.destination, force=force, quiet=quiet)

    unresolved = [target.name for target in missing_targets if not target.is_installed()]
    if unresolved:
        raise AssetSetupError(
            "Asset installation completed, but these targets are still missing: "
            + ", ".join(sorted(unresolved))
        )

    if not quiet:
        print("[INFO] Asset setup completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and install PatchEX-Design and PED-Eval assets into the expected directories.",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        choices=sorted(ZENODO_RECORDS.keys()),
        help="Asset bundle to prepare. Repeat to install multiple bundles. Defaults to all bundles.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root where assets should be installed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall assets even if the expected files already exist.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        ensure_assets(
            bundles=args.bundle,
            repo_root=args.repo_root,
            force=args.force,
            quiet=args.quiet,
        )
    except AssetSetupError as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
