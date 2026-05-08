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
from typing import Callable, Iterable, Optional, Set, Union


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
REQUIRED_DATA_DESIGN_PATHS = {
    "opt.csv",
    "ph.csv",
    "pdb",
    "ec_pools",
    "functional_site_annotations",
    "functional_site_annotations/opt/functional_sites_long.csv",
    "functional_site_annotations/ph/functional_sites_long.csv",
}
REQUIRED_PREDICTOR_WEIGHT_ITEMS = {
    "ephod",
    "patchex_weight",
    "patchet_pretrain_weight",
}
REQUIRED_PREDICTOR_WEIGHT_PATHS = {
    "ephod/ESM1v-SVR.pkl",
    "ephod/model_pHopt_window.3_r2.0.42.pth",
    "ephod/model_topt_window.3_r2.0.57.pth",
    "patchex_weight/opt/model_config.yaml",
    "patchex_weight/opt/model.safetensors",
    "patchex_weight/ph/model_config.yaml",
    "patchex_weight/ph/model.safetensors",
    "patchet_pretrain_weight/model_config.yaml",
    "patchet_pretrain_weight/model.safetensors",
}
REQUIRED_PIPELINE_WEIGHT_ITEMS = {"opt", "ph"}
REQUIRED_PIPELINE_WEIGHT_PATHS = {
    "opt/model_config.yaml",
    "opt/model.safetensors",
    "ph/model_config.yaml",
    "ph/model.safetensors",
}
REQUIRED_ESM150_PATHS = {
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
}
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


class AssetSetupError(RuntimeError):
    pass


def _format_size(num_bytes: int) -> str:
    size = float(max(num_bytes, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


class _ProgressBar:
    def __init__(self, label: str, *, total: Optional[int] = None, stream=None) -> None:
        self.label = label
        self.total = total if total and total > 0 else None
        self.stream = stream or sys.stderr
        self.width = 32
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._last_percent = -1
        self._last_text = ""

    def update(self, completed: int) -> None:
        text = self._render(completed)
        if text == self._last_text:
            return
        self._last_text = text
        if self.is_tty:
            self.stream.write(f"\r{text}")
        else:
            self.stream.write(f"{text}\n")
        self.stream.flush()

    def close(self, completed: int) -> None:
        text = self._render(completed, final=True)
        if text != self._last_text:
            if self.is_tty:
                self.stream.write(f"\r{text}")
            else:
                self.stream.write(f"{text}\n")
            self.stream.flush()
        if self.is_tty:
            self.stream.write("\n")
            self.stream.flush()

    def _render(self, completed: int, *, final: bool = False) -> str:
        completed = max(completed, 0)
        if self.total is None:
            return f"[INFO] {self.label}: {_format_size(completed)}"

        ratio = min(completed / self.total, 1.0)
        percent = int(ratio * 100)
        if not self.is_tty and not final and percent < 100:
            percent_bucket = percent // 10
            if percent_bucket == self._last_percent:
                return self._last_text
            self._last_percent = percent_bucket
        filled = min(self.width, int(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        return (
            f"[INFO] {self.label}: [{bar}] {percent:3d}% "
            f"({_format_size(completed)}/{_format_size(self.total)})"
        )


@dataclass(frozen=True)
class AssetTarget:
    bundle: str
    name: str
    destination: Path
    kind: str
    validator: Callable[[Path], bool]
    finder: Callable[[Path], Optional[Path]]

    def is_installed(self) -> bool:
        return self.validator(self.destination)


def _directory_with_children_validator(required_children: set[str]) -> Callable[[Path], bool]:
    def _validator(path: Path) -> bool:
        if not path.is_dir():
            return False
        child_names = {child.name for child in path.iterdir()}
        return required_children.issubset(child_names)

    return _validator


def _directory_with_required_paths_validator(required_paths: set[str]) -> Callable[[Path], bool]:
    normalized_paths = tuple(Path(relative_path) for relative_path in sorted(required_paths))

    def _validator(path: Path) -> bool:
        if not path.is_dir():
            return False
        return all((path / relative_path).exists() for relative_path in normalized_paths)

    return _validator


def _existing_directory(path: Path) -> bool:
    return path.is_dir()


def _existing_file(path: Path) -> bool:
    return path.is_file()


def _find_named_path(stage_root: Path, name: str, *, kind: str = "dir", exclude_parts: Optional[Set[str]] = None) -> Optional[Path]:
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


def _find_parent_with_children(stage_root: Path, required_children: Set[str], *, exclude_parts: Optional[Set[str]] = None) -> Optional[Path]:
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
            validator=_directory_with_required_paths_validator(REQUIRED_DATA_DESIGN_PATHS),
            finder=lambda stage_root: _find_named_path(stage_root, "data_design", kind="dir")
            or _find_parent_with_children(stage_root, REQUIRED_DATA_DESIGN_ITEMS),
        ),
        "predictor_weights": AssetTarget(
            bundle="ped-eval",
            name="predictor_weights",
            destination=repo_root / "predictor_weights",
            kind="dir",
            validator=_directory_with_required_paths_validator(REQUIRED_PREDICTOR_WEIGHT_PATHS),
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
            validator=_directory_with_required_paths_validator(REQUIRED_PIPELINE_WEIGHT_PATHS),
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
            validator=_directory_with_required_paths_validator(REQUIRED_ESM150_PATHS),
            finder=lambda stage_root: _find_named_path(stage_root, "esm150", kind="dir"),
        ),
    }


def _download_file(url: str, destination: Path, *, quiet: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"[INFO] Downloading {url} -> {destination}")
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as output_file:
            total_size: Optional[int] = None
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    total_size = int(content_length)
                except ValueError:
                    total_size = None

            progress = None if quiet else _ProgressBar(f"Downloading {destination.name}", total=total_size)
            bytes_written = 0
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                output_file.write(chunk)
                bytes_written += len(chunk)
                if progress is not None:
                    progress.update(bytes_written)
            if progress is not None:
                progress.close(bytes_written)
    except urllib.error.URLError as exc:
        raise AssetSetupError(f"Failed to download {url}: {exc}") from exc
    return destination


def _record_file_matches_target(file_entry: dict, target_names: set[str]) -> bool:
    raw_file_name = str(file_entry.get("key") or file_entry.get("filename") or "")
    if not raw_file_name:
        return False

    normalized_name = raw_file_name.lower()
    base_name = Path(normalized_name).name
    for target_name in target_names:
        normalized_target = str(target_name).lower()
        target_stem = Path(normalized_target).stem
        if normalized_target in normalized_name or normalized_target in base_name:
            return True
        if target_stem and (target_stem in normalized_name or target_stem in base_name):
            return True
    return False


def _fetch_zenodo_record_files(
    record_id: int,
    download_dir: Path,
    *,
    quiet: bool = False,
    target_names: Optional[Set[str]] = None,
) -> list[Path]:
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

    selected_files = files
    if target_names:
        matched_files = [file_entry for file_entry in files if _record_file_matches_target(file_entry, target_names)]
        if matched_files:
            selected_files = matched_files
            if not quiet:
                matched_names = ", ".join(sorted(target_names))
                print(f"[INFO] Limiting Zenodo record {record_id} downloads to targets: {matched_names}")

    downloaded_files: list[Path] = []
    for file_entry in selected_files:
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
        if zipfile.is_zipfile(file_path):
            if not quiet:
                print(f"[INFO] Extracting zip archive {file_path.name}")
            with zipfile.ZipFile(file_path) as archive:
                archive.extractall(stage_root)  # ← FIX: extract directly
            continue

        if tarfile.is_tarfile(file_path):
            if not quiet:
                print(f"[INFO] Extracting tar archive {file_path.name}")
            with tarfile.open(file_path) as archive:
                archive.extractall(stage_root)  # ← FIX
            continue

        # fallback for non-archive files
        shutil.copy2(file_path, stage_root / file_path.name)


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
    bundles: Optional[Iterable[str]] = None,
    *,
    repo_root: Optional[Union[str, Path]] = None,
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

    record_ids = list(dict.fromkeys(ZENODO_RECORDS[target.bundle] for target in missing_targets))
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
            target_names = {
                target.name
                for target in missing_targets
                if ZENODO_RECORDS[target.bundle] == record_id
            }
            downloaded_files = _fetch_zenodo_record_files(
                record_id,
                download_dir,
                quiet=quiet,
                target_names=target_names,
            )
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
