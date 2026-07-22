from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

QWEN_ASR_MODEL = "qwen3-asr-0.6b-4bit"
QWEN_FORCE_MODEL = "qwen3-forced-aligner-0.6b-4bit"
QWEN_MODEL_NAMES = (QWEN_ASR_MODEL, QWEN_FORCE_MODEL)
MARKER_FILE = ".cutvideo-qwen-models.json"
_VERIFIED_ROOT_SIGNATURES: dict[str, tuple[object, ...]] = {}


class ResourceError(RuntimeError):
    """Raised when an offline runtime resource is missing or unsupported."""


class ModelPreparationError(ResourceError):
    """Raised when Qwen model preparation fails."""


class ModelPreparationCancelled(ModelPreparationError):
    """Raised when Qwen model preparation is cancelled by the caller."""


@dataclass(frozen=True, slots=True)
class ModelPreparationResult:
    model_root: Path
    source: str
    downloaded: bool = False
    migrated: bool = False


ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]


def default_model_root() -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "CutVideo"
        / "models"
    )


def system_model_root() -> Path:
    """Return the read-only store populated by the macOS upgrade installer."""

    return Path("/Library/Application Support/CutVideo/models")


def _platform_key() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "macos-arm64"
    return f"{sys.platform}-{machine}"


def _resource_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    if value := os.environ.get("CUTVIDEO_RESOURCE_ROOT"):
        candidates.append(Path(value).expanduser())
    if frozen_root := getattr(sys, "_MEIPASS", None):
        candidates.append(Path(frozen_root) / "resources")

    executable_dir = Path(sys.executable).resolve().parent
    candidates.extend((executable_dir / "resources", executable_dir.parent / "Resources" / "resources"))

    source_root = Path(__file__).resolve().parents[2]
    candidates.append(source_root / "resources")
    return candidates


def _find_resource_root() -> Path:
    for candidate in _resource_root_candidates():
        if (candidate / "manifest.json").is_file():
            return candidate.resolve()
    return _resource_root_candidates()[0].resolve()


def _load_manifest(resource_root: Path) -> dict[str, object]:
    manifest_path = resource_root / "manifest.json"
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPreparationError(f"无法读取模型清单：{manifest_path}") from exc


def _model_entry(manifest: dict[str, object], name: str) -> dict[str, object]:
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise ModelPreparationError("模型清单格式无效：缺少 models")
    entry = models.get(name)
    if not isinstance(entry, dict):
        raise ModelPreparationError(f"模型清单格式无效：缺少 {name}")
    return entry


def _expected_tree_sha(entry: dict[str, object], model_name: str) -> str:
    value = entry.get("sha256")
    if not isinstance(value, str) or not value:
        raise ModelPreparationError(f"模型清单格式无效：{model_name} 缺少 sha256")
    return value.lower()


def _required_files(entry: dict[str, object]) -> list[str]:
    value = entry.get("download", {})
    files = value.get("required_files") if isinstance(value, dict) else None
    if isinstance(files, list) and all(isinstance(item, str) and item for item in files):
        return files
    return [
        "config.json",
        "merges.txt",
        "quantization_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "weights.safetensors",
    ]


def _sha256_file(path: Path, cancelled: CancelCallback | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            _check_cancelled(cancelled)
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(directory: Path, cancelled: CancelCallback | None = None) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        _check_cancelled(cancelled)
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                _check_cancelled(cancelled)
                digest.update(chunk)
    return digest.hexdigest()


def _marker_path(model_root: Path) -> Path:
    return model_root / MARKER_FILE


def _read_marker(model_root: Path) -> dict[str, object] | None:
    try:
        data = json.loads(_marker_path(model_root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _file_stamps(model_dir: Path) -> dict[str, dict[str, int]]:
    stamps: dict[str, dict[str, int]] = {}
    for path in sorted(p for p in model_dir.rglob("*") if p.is_file()):
        info = path.stat()
        stamps[path.relative_to(model_dir).as_posix()] = {
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
        }
    return stamps


def _root_signature(model_root: Path, manifest: dict[str, object]) -> tuple[object, ...] | None:
    signature: list[object] = []
    try:
        for name in QWEN_MODEL_NAMES:
            entry = _model_entry(manifest, name)
            model_dir = model_root / name
            if not _has_required_files(model_dir, entry):
                return None
            signature.extend((name, _expected_tree_sha(entry, name)))
            for relative, stamp in sorted(_file_stamps(model_dir).items()):
                signature.extend((relative, stamp["size"], stamp["mtime_ns"]))
    except (OSError, ModelPreparationError):
        return None
    return tuple(signature)


def _remember_verified_root(model_root: Path, manifest: dict[str, object]) -> None:
    signature = _root_signature(model_root, manifest)
    if signature is not None:
        _VERIFIED_ROOT_SIGNATURES[str(model_root.expanduser().resolve())] = signature


def _root_was_verified(model_root: Path, manifest: dict[str, object]) -> bool:
    key = str(model_root.expanduser().resolve())
    remembered = _VERIFIED_ROOT_SIGNATURES.get(key)
    return remembered is not None and remembered == _root_signature(model_root, manifest)


def _marker_files_match(
    model_root: Path,
    manifest: dict[str, object],
    cancelled: CancelCallback | None = None,
) -> bool:
    try:
        for name in QWEN_MODEL_NAMES:
            entry = _model_entry(manifest, name)
            download = entry.get("download")
            expected_files = download.get("file_sha256") if isinstance(download, dict) else None
            required = set(_required_files(entry))
            if not isinstance(expected_files, dict) or set(expected_files) != required:
                return False
            for relative in sorted(required):
                expected = expected_files.get(relative)
                if not isinstance(expected, str) or not expected:
                    return False
                actual = _sha256_file(model_root / name / relative, cancelled)
                if actual.lower() != expected.lower():
                    return False
    except OSError:
        return False
    return True


def _marker_model_is_valid(
    model_root: Path,
    model_name: str,
    entry: dict[str, object],
) -> bool:
    marker = _read_marker(model_root)
    if not marker:
        return False
    models = marker.get("models")
    if not isinstance(models, dict):
        return False
    record = models.get(model_name)
    if not isinstance(record, dict):
        return False
    if str(record.get("sha256", "")).lower() != _expected_tree_sha(entry, model_name):
        return False
    files = record.get("files")
    if not isinstance(files, dict):
        return False
    model_dir = model_root / model_name
    if not model_dir.is_dir():
        return False
    required = set(_required_files(entry))
    if not required.issubset(files):
        return False
    current_files = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file()
    }
    if current_files != set(files):
        return False
    for relative, stamp in files.items():
        if not isinstance(relative, str) or not isinstance(stamp, dict):
            return False
        path = model_dir / relative
        try:
            info = path.stat()
        except OSError:
            return False
        if info.st_size != stamp.get("size") or info.st_mtime_ns != stamp.get("mtime_ns"):
            return False
    return True


def _write_marker(model_root: Path, manifest: dict[str, object]) -> None:
    models: dict[str, object] = {}
    for name in QWEN_MODEL_NAMES:
        entry = _model_entry(manifest, name)
        model_dir = model_root / name
        models[name] = {
            "sha256": _expected_tree_sha(entry, name),
            "files": _file_stamps(model_dir),
        }
    payload = {
        "schema_version": 1,
        "models": models,
    }
    marker = _marker_path(model_root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_name(f"{marker.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, marker)


def _has_required_files(model_dir: Path, entry: dict[str, object]) -> bool:
    return model_dir.is_dir() and all((model_dir / file).is_file() for file in _required_files(entry))


def _verify_model_dir(
    model_root: Path,
    model_name: str,
    entry: dict[str, object],
    *,
    full: bool,
    cancelled: CancelCallback | None = None,
) -> bool:
    model_dir = model_root / model_name
    if not _has_required_files(model_dir, entry):
        return False
    if not full:
        return True

    weight_sha = None
    download = entry.get("download")
    if isinstance(download, dict):
        value = download.get("weight_sha256")
        weight_sha = value.lower() if isinstance(value, str) else None
    if (
        weight_sha
        and _sha256_file(model_dir / "weights.safetensors", cancelled).lower() != weight_sha
    ):
        return False
    return _sha256_tree(model_dir, cancelled).lower() == _expected_tree_sha(entry, model_name)


def qwen_models_ready(model_root: Path, manifest_root: Path, full: bool = False) -> bool:
    manifest = _load_manifest(manifest_root)
    model_root = Path(model_root).expanduser()
    if _root_was_verified(model_root, manifest):
        return True
    if full:
        try:
            ok = all(
                _verify_model_dir(model_root, name, _model_entry(manifest, name), full=True)
                for name in QWEN_MODEL_NAMES
            )
            if ok:
                with contextlib.suppress(OSError):
                    _write_marker(model_root, manifest)
                _remember_verified_root(model_root, manifest)
            return ok
        except (OSError, ModelPreparationError):
            return False

    try:
        marker_valid = all(
            _marker_model_is_valid(model_root, name, _model_entry(manifest, name))
            for name in QWEN_MODEL_NAMES
        )
        if not marker_valid or not _marker_files_match(model_root, manifest):
            return False
        _remember_verified_root(model_root, manifest)
        return True
    except ModelPreparationCancelled:
        raise
    except (OSError, ModelPreparationError):
        return False


def _qwen_models_marker_ready(
    model_root: Path,
    manifest: dict[str, object],
    cancelled: CancelCallback | None = None,
) -> bool:
    if _root_was_verified(model_root, manifest):
        return True
    marker_valid = all(
        _marker_model_is_valid(model_root, name, _model_entry(manifest, name))
        for name in QWEN_MODEL_NAMES
    )
    if not marker_valid or not _marker_files_match(model_root, manifest, cancelled):
        return False
    _remember_verified_root(model_root, manifest)
    return True


def _qwen_models_fully_ready(
    model_root: Path,
    manifest: dict[str, object],
    cancelled: CancelCallback | None = None,
) -> bool:
    if _root_was_verified(model_root, manifest):
        return True
    try:
        ok = all(
            _verify_model_dir(
                model_root,
                name,
                _model_entry(manifest, name),
                full=True,
                cancelled=cancelled,
            )
            for name in QWEN_MODEL_NAMES
        )
        if ok:
            with contextlib.suppress(OSError):
                _write_marker(model_root, manifest)
            _remember_verified_root(model_root, manifest)
        return ok
    except ModelPreparationCancelled:
        raise
    except (OSError, ModelPreparationError):
        return False


def _emit(progress: ProgressCallback | None, fraction: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, fraction)), message)


def _check_cancelled(cancelled: CancelCallback | None) -> None:
    if cancelled and cancelled():
        raise ModelPreparationCancelled("已取消模型准备")


def _copytree_checked(
    source: Path,
    target: Path,
    cancelled: CancelCallback | None,
    progress: ProgressCallback | None,
    start: float,
    end: float,
    label: str,
) -> None:
    files = [path for path in source.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files) or 1
    copied = 0
    target.mkdir(parents=True, exist_ok=True)
    for source_file in files:
        _check_cancelled(cancelled)
        relative = source_file.relative_to(source)
        target_file = target / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with source_file.open("rb") as source_stream, target_file.open("wb") as target_stream:
            while True:
                _check_cancelled(cancelled)
                chunk = source_stream.read(1024 * 1024)
                if not chunk:
                    break
                target_stream.write(chunk)
                copied += len(chunk)
                _emit(progress, start + (end - start) * copied / total, label)
        shutil.copystat(source_file, target_file)


def _legacy_bundle_model_roots(resource_root: Path) -> list[Path]:
    candidates = [
        resource_root / "models",
        system_model_root(),
        Path("/Applications/CutVideo.app/Contents/Resources/resources/models"),
        Path.home() / "Applications/CutVideo.app/Contents/Resources/resources/models",
    ]
    volumes = Path("/Volumes")
    if volumes.is_dir():
        with contextlib.suppress(OSError):
            candidates.extend(
                app / "Contents/Resources/resources/models"
                for app in volumes.glob("*/CutVideo.app")
            )

    seen: set[Path] = set()
    result: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved not in seen and resolved.is_dir():
            seen.add(resolved)
            result.append(resolved)
    return result


def _download_entry(entry: dict[str, object], model_name: str) -> dict[str, object]:
    download = entry.get("download")
    if not isinstance(download, dict):
        raise ModelPreparationError(f"模型清单缺少下载信息：{model_name}")
    for key in ("url", "archive_sha256", "archive_size"):
        if key not in download:
            raise ModelPreparationError(f"模型清单下载信息不完整：{model_name} 缺少 {key}")
    if not isinstance(download.get("url"), str) or not isinstance(download.get("archive_sha256"), str):
        raise ModelPreparationError(f"模型清单下载信息格式无效：{model_name}")
    if not isinstance(download.get("archive_size"), int):
        raise ModelPreparationError(f"模型清单下载大小格式无效：{model_name}")
    return download


def _download_archive(
    model_name: str,
    entry: dict[str, object],
    cache_dir: Path,
    cancelled: CancelCallback | None,
    progress: ProgressCallback | None,
    start: float,
    end: float,
) -> Path:
    download = _download_entry(entry, model_name)
    url = str(download["url"])
    expected_sha = str(download["archive_sha256"]).lower()
    expected_size = int(download["archive_size"])
    archive = cache_dir / f"{model_name}.zip"
    partial = cache_dir / f"{model_name}.zip.part"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if archive.is_file() and archive.stat().st_size == expected_size:
        if _sha256_file(archive, cancelled).lower() == expected_sha:
            _emit(progress, end, f"{model_name} 已有下载缓存")
            return archive
        archive.unlink()

    if partial.is_file() and partial.stat().st_size == expected_size:
        if _sha256_file(partial, cancelled).lower() == expected_sha:
            os.replace(partial, archive)
            _emit(progress, end, f"{model_name} 下载完成")
            return archive
        partial.unlink()

    resume_from = partial.stat().st_size if partial.is_file() else 0
    if resume_from > expected_size:
        partial.unlink(missing_ok=True)
        resume_from = 0
    request = urllib.request.Request(url)
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")

    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.URLError as exc:
        raise ModelPreparationError(f"下载模型失败：{model_name}，{exc}") from exc

    with response:
        status = getattr(response, "status", None)
        if resume_from and status != 206:
            partial.unlink(missing_ok=True)
            return _download_archive(
                model_name, entry, cache_dir, cancelled, progress, start, end
            )

        mode = "ab" if resume_from else "wb"
        downloaded = resume_from
        with partial.open(mode) as stream:
            while True:
                _check_cancelled(cancelled)
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                downloaded += len(chunk)
                if expected_size:
                    _emit(
                        progress,
                        start + (end - start) * min(downloaded, expected_size) / expected_size,
                        f"正在下载 {model_name}",
                    )

    if partial.stat().st_size != expected_size:
        raise ModelPreparationError(
            f"模型下载不完整：{model_name}，期望 {expected_size} 字节，实际 {partial.stat().st_size} 字节"
        )
    if _sha256_file(partial, cancelled).lower() != expected_sha:
        partial.unlink(missing_ok=True)
        raise ModelPreparationError(f"模型压缩包校验失败：{model_name}")
    os.replace(partial, archive)
    _emit(progress, end, f"{model_name} 下载完成")
    return archive


def _safe_extract_model_archive(
    archive: Path,
    model_name: str,
    entry: dict[str, object],
    target: Path,
    cancelled: CancelCallback | None,
) -> None:
    scratch = target.parent / f".extract-{model_name}-{os.getpid()}-{uuid.uuid4().hex}"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zip_file:
            for member in zip_file.infolist():
                _check_cancelled(cancelled)
                name = member.filename
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ModelPreparationError(f"模型压缩包包含不安全路径：{model_name}")
                mode = (member.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise ModelPreparationError(f"模型压缩包包含不支持的符号链接：{model_name}")
                destination = scratch / name
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member) as source, destination.open("wb") as output:
                    while True:
                        _check_cancelled(cancelled)
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)

        required = _required_files(entry)
        if all((scratch / file).is_file() for file in required):
            source_dir = scratch
        elif (scratch / model_name).is_dir() and all(
            (scratch / model_name / file).is_file() for file in required
        ):
            source_dir = scratch / model_name
        else:
            children = [child for child in scratch.iterdir() if child.is_dir()]
            source_dir = next(
                (
                    child
                    for child in children
                    if all((child / file).is_file() for file in required)
                ),
                None,
            )
            if source_dir is None:
                raise ModelPreparationError(f"模型压缩包内容不完整：{model_name}")

        if target.exists():
            shutil.rmtree(target)
        # scratch 与 target 位于同一文件系统，直接移动可避免再复制一份
        # 约 0.5 GB 的模型和复制阶段的长时间不可取消等待。
        os.replace(source_dir, target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@contextlib.contextmanager
def _model_lock(model_root: Path, cancelled: CancelCallback | None) -> Iterator[None]:
    lock_parent = model_root.parent
    lock_parent.mkdir(parents=True, exist_ok=True)
    lock_path = lock_parent / ".cutvideo-models.lock"
    stream = lock_path.open("a+")
    try:
        try:
            import fcntl
        except ImportError:
            yield
            return
        while True:
            _check_cancelled(cancelled)
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.2)
        yield
    finally:
        with contextlib.suppress(Exception):
            if "fcntl" in sys.modules:
                sys.modules["fcntl"].flock(stream.fileno(), sys.modules["fcntl"].LOCK_UN)
        stream.close()


def _publish_model_root(stage_root: Path, model_root: Path) -> None:
    model_root.parent.mkdir(parents=True, exist_ok=True)
    backup = model_root.with_name(f"{model_root.name}.backup-{os.getpid()}-{uuid.uuid4().hex}")
    moved_existing = False
    try:
        if model_root.exists():
            os.replace(model_root, backup)
            moved_existing = True
        os.replace(stage_root, model_root)
        if moved_existing:
            shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        if model_root.exists() and model_root != stage_root:
            shutil.rmtree(model_root, ignore_errors=True)
        if moved_existing and backup.exists():
            os.replace(backup, model_root)
        raise


def _prepare_from_legacy_bundle(
    source_root: Path,
    target_root: Path,
    manifest: dict[str, object],
    cancelled: CancelCallback | None,
    progress: ProgressCallback | None,
) -> bool:
    if not all(
        _verify_model_dir(
            source_root,
            name,
            _model_entry(manifest, name),
            full=True,
            cancelled=cancelled,
        )
        for name in QWEN_MODEL_NAMES
    ):
        return False
    stage_root = target_root.parent / f".models-stage-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        for index, name in enumerate(QWEN_MODEL_NAMES):
            start = 0.1 + 0.35 * index
            end = start + 0.35
            _copytree_checked(
                source_root / name,
                stage_root / name,
                cancelled,
                progress,
                start,
                end,
                f"正在迁移旧版模型 {name}",
            )
        if not all(
            _verify_model_dir(
                stage_root,
                name,
                _model_entry(manifest, name),
                full=True,
                cancelled=cancelled,
            )
            for name in QWEN_MODEL_NAMES
        ):
            raise ModelPreparationError("旧版模型迁移后校验失败")
        _write_marker(stage_root, manifest)
        _publish_model_root(stage_root, target_root)
        _remember_verified_root(target_root, manifest)
        return True
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root, ignore_errors=True)


def _manifest_with_root(resource_root: Path) -> dict[str, object]:
    manifest = _load_manifest(resource_root)
    manifest["_manifest_root"] = str(resource_root)
    return manifest


def _prepare_from_download(
    target_root: Path,
    manifest: dict[str, object],
    cancelled: CancelCallback | None,
    progress: ProgressCallback | None,
) -> None:
    cache_dir = target_root.parent / "model-download-cache"
    stage_root = target_root.parent / f".models-stage-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        stage_root.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(QWEN_MODEL_NAMES):
            entry = _model_entry(manifest, name)
            archive = _download_archive(
                name,
                entry,
                cache_dir,
                cancelled,
                progress,
                0.05 + index * 0.4,
                0.35 + index * 0.4,
            )
            _check_cancelled(cancelled)
            _emit(progress, 0.36 + index * 0.4, f"正在解压 {name}")
            _safe_extract_model_archive(
                archive,
                name,
                entry,
                stage_root / name,
                cancelled,
            )
            _emit(progress, 0.45 + index * 0.4, f"正在校验 {name}")
            if not _verify_model_dir(
                stage_root,
                name,
                entry,
                full=True,
                cancelled=cancelled,
            ):
                raise ModelPreparationError(f"模型解压后校验失败：{name}")
        _write_marker(stage_root, manifest)
        _publish_model_root(stage_root, target_root)
        _remember_verified_root(target_root, manifest)
        # Resumable archives are useful only until the atomic model store is
        # published. Removing them here avoids keeping a second ~1.06 GB copy
        # beside the installed models.
        for name in QWEN_MODEL_NAMES:
            with contextlib.suppress(OSError):
                (cache_dir / f"{name}.zip").unlink()
        with contextlib.suppress(OSError):
            cache_dir.rmdir()
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root, ignore_errors=True)


def _explicit_model_paths() -> tuple[Path, Path] | None:
    asr = os.environ.get("CUTVIDEO_QWEN_ASR_MODEL")
    force = os.environ.get("CUTVIDEO_QWEN_FORCE_MODEL")
    if not asr and not force:
        return None
    if not asr or not force:
        raise ModelPreparationError(
            "显式模型路径不完整：CUTVIDEO_QWEN_ASR_MODEL 和 CUTVIDEO_QWEN_FORCE_MODEL 需要同时设置"
        )
    return Path(asr).expanduser(), Path(force).expanduser()


def _verify_explicit_model_paths(
    asr_path: Path,
    force_path: Path,
    manifest: dict[str, object],
    cancelled: CancelCallback | None = None,
) -> bool:
    for model_name, path in ((QWEN_ASR_MODEL, asr_path), (QWEN_FORCE_MODEL, force_path)):
        entry = _model_entry(manifest, model_name)
        if path.name != model_name:
            ok = _has_required_files(path, entry) and _sha256_tree(
                path,
                cancelled,
            ).lower() == _expected_tree_sha(entry, model_name)
        else:
            ok = _verify_model_dir(
                path.parent,
                model_name,
                entry,
                full=True,
                cancelled=cancelled,
            )
        if not ok:
            return False
    return True


def prepare_qwen_models(
    *,
    cancelled: CancelCallback | None = None,
    progress: ProgressCallback | None = None,
    resource_root: Path | None = None,
    target_root: Path | None = None,
) -> ModelPreparationResult:
    if _platform_key() != "macos-arm64":
        raise ModelPreparationError("当前平台不支持 Qwen MLX 模型，只支持 Apple Silicon macOS")

    root = (resource_root or _find_resource_root()).expanduser().resolve()
    manifest = _manifest_with_root(root)
    _check_cancelled(cancelled)

    custom_target = target_root is not None
    explicit_paths = _explicit_model_paths() if not custom_target else None
    if explicit_paths is not None:
        _emit(progress, 0.1, "正在校验显式 Qwen 模型路径")
        if not _verify_explicit_model_paths(*explicit_paths, manifest, cancelled):
            raise ModelPreparationError("显式 Qwen 模型路径校验失败")
        _emit(progress, 1.0, "显式 Qwen 模型可用")
        return ModelPreparationResult(
            model_root=explicit_paths[0].parent,
            source="env",
        )

    explicit_root = os.environ.get("CUTVIDEO_MODEL_ROOT") if not custom_target else None
    selected_target = (
        Path(target_root).expanduser()
        if target_root is not None
        else Path(explicit_root).expanduser()
        if explicit_root
        else default_model_root()
    )
    selected_target = selected_target.resolve()
    if explicit_root:
        _emit(progress, 0.1, "正在校验 CUTVIDEO_MODEL_ROOT")
        if not _qwen_models_fully_ready(selected_target, manifest, cancelled):
            raise ModelPreparationError(
                f"CUTVIDEO_MODEL_ROOT 中缺少可用 Qwen 模型：{selected_target}"
            )
        _emit(progress, 1.0, "CUTVIDEO_MODEL_ROOT Qwen 模型可用")
        return ModelPreparationResult(model_root=selected_target, source="env")

    _emit(progress, 0.02, "正在校验本机 Qwen 模型…")
    if _qwen_models_marker_ready(selected_target, manifest, cancelled) or _qwen_models_fully_ready(
        selected_target,
        manifest,
        cancelled,
    ):
        _emit(progress, 1.0, "本机 Qwen 模型已准备好")
        return ModelPreparationResult(model_root=selected_target, source="external")

    installer_models = system_model_root()
    if not custom_target and selected_target != installer_models:
        _emit(progress, 0.03, "正在校验旧版安装模型…")
        if _qwen_models_fully_ready(installer_models, manifest, cancelled):
            _emit(progress, 1.0, "旧版 Qwen 模型可直接使用")
            return ModelPreparationResult(
                model_root=installer_models,
                source="installer-migrated",
                migrated=True,
            )

    with _model_lock(selected_target, cancelled):
        _check_cancelled(cancelled)
        if _qwen_models_marker_ready(
            selected_target,
            manifest,
            cancelled,
        ) or _qwen_models_fully_ready(
            selected_target,
            manifest,
            cancelled,
        ):
            _emit(progress, 1.0, "本机 Qwen 模型已准备好")
            return ModelPreparationResult(model_root=selected_target, source="external")

        if not custom_target and selected_target != installer_models and _qwen_models_fully_ready(
            installer_models,
            manifest,
            cancelled,
        ):
            _emit(progress, 1.0, "旧版 Qwen 模型可直接使用")
            return ModelPreparationResult(
                model_root=installer_models,
                source="installer-migrated",
                migrated=True,
            )

        for candidate in _legacy_bundle_model_roots(root):
            _check_cancelled(cancelled)
            if candidate.resolve() == selected_target:
                continue
            if not custom_target and candidate.resolve() == installer_models:
                continue
            _emit(progress, 0.05, f"正在检查旧版模型：{candidate}")
            if _prepare_from_legacy_bundle(
                candidate,
                selected_target,
                manifest,
                cancelled,
                progress,
            ):
                _emit(progress, 1.0, "旧版 Qwen 模型已迁移")
                return ModelPreparationResult(
                    model_root=selected_target,
                    source="migrated",
                    migrated=True,
                )

        _emit(progress, 0.05, "本机没有可迁移模型，开始下载 Qwen 模型")
        _prepare_from_download(selected_target, manifest, cancelled, progress)
        _emit(progress, 1.0, "Qwen 模型下载完成")
        return ModelPreparationResult(
            model_root=selected_target,
            source="downloaded",
            downloaded=True,
        )
