"""Small, dependency-free macOS window chrome helpers."""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

_FULL_SIZE_CONTENT_VIEW = 1 << 15
_TITLE_HIDDEN = 1
_TITLEBAR_SEPARATOR_NONE = 0


class WindowDragRegion(QWidget):
    """Transparent client-area strip that preserves native window dragging."""

    def __init__(self, parent: QWidget | None = None, *, height: int = 28) -> None:
        super().__init__(parent)
        self.setObjectName("windowDragRegion")
        self.setFixedHeight(height if sys.platform == "darwin" else 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._drag_offset: QPoint | None = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        window = self.window()
        handle = window.windowHandle()
        if handle is not None and handle.startSystemMove():
            event.accept()
            return
        self._drag_offset = (
            event.globalPosition().toPoint() - window.frameGeometry().topLeft()
        )
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.window().move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)


def make_titlebar_immersive(widget: QWidget) -> bool:
    """Extend Qt content below the native traffic lights and hide the title."""

    if sys.platform != "darwin":
        return False

    runtime_path = ctypes.util.find_library("objc")
    if not runtime_path:
        return False

    try:
        runtime = ctypes.CDLL(runtime_path)
        runtime.sel_registerName.restype = ctypes.c_void_p
        runtime.sel_registerName.argtypes = (ctypes.c_char_p,)
        send_address = ctypes.cast(runtime.objc_msgSend, ctypes.c_void_p).value
        if not send_address:
            return False

        def selector(name: str):
            return runtime.sel_registerName(name.encode("ascii"))

        send_pointer = ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )(send_address)
        send_integer = ctypes.CFUNCTYPE(
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )(send_address)
        send_bool_selector = ctypes.CFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )(send_address)
        send_bool = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_bool,
        )(send_address)
        send_ulong = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        )(send_address)

        native_view = int(widget.winId())
        native_window = send_pointer(native_view, selector("window"))
        if not native_window:
            return False

        style_mask = send_integer(native_window, selector("styleMask"))
        send_ulong(
            native_window,
            selector("setStyleMask:"),
            style_mask | _FULL_SIZE_CONTENT_VIEW,
        )
        send_bool(
            native_window,
            selector("setTitlebarAppearsTransparent:"),
            True,
        )
        send_ulong(native_window, selector("setTitleVisibility:"), _TITLE_HIDDEN)
        send_bool(
            native_window,
            selector("setMovableByWindowBackground:"),
            True,
        )

        separator_selector = selector("setTitlebarSeparatorStyle:")
        if send_bool_selector(
            native_window,
            selector("respondsToSelector:"),
            separator_selector,
        ):
            send_ulong(
                native_window,
                separator_selector,
                _TITLEBAR_SEPARATOR_NONE,
            )
        return True
    except (AttributeError, OSError, TypeError, ValueError):
        # Keep the normal Qt title bar if the native bridge is unavailable.
        return False


__all__ = ["WindowDragRegion", "make_titlebar_immersive"]
