from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, QSize
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parents[1]
ICON_ROOT = ROOT / "resources" / "icons"


def render_icon(size: int) -> QImage:
    renderer = QSvgRenderer(str(ICON_ROOT / "app-icon.svg"))
    if not renderer.isValid():
        raise RuntimeError("invalid app-icon.svg")
    image = QImage(QSize(size, size), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return image


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
