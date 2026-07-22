"""Preserve MLX's native libraries and compiled Metal shader bundle."""

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

binaries = collect_dynamic_libs("mlx", destdir="mlx/lib")
datas = collect_data_files("mlx", includes=["lib/mlx.metallib"])
hiddenimports = collect_submodules("mlx")
