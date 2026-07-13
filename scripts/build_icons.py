from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QGuiApplication, QImage

ROOT = Path(__file__).resolve().parents[1]
ICON_ROOT = ROOT / "resources" / "icons"
ICON_SOURCE = ICON_ROOT / "app-icon-source.png"


def render_icon(size: int) -> QImage:
    source = QImage(str(ICON_SOURCE))
    if source.isNull():
        raise RuntimeError(f"invalid app icon source: {ICON_SOURCE}")
    return source.scaled(
        QSize(size, size),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def main() -> int:
    QGuiApplication.instance() or QGuiApplication(["build-icons"])
    ICON_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "app-icon.png": render_icon(512),
        "app-icon.ico": render_icon(256),
        "app-icon.icns": render_icon(1024),
    }
    for name, image in outputs.items():
        path = ICON_ROOT / name
        if not image.save(str(path)):
            raise RuntimeError(f"cannot write {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
