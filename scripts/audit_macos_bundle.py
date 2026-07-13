from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def _version(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit every Mach-O embedded in a macOS app")
    parser.add_argument("app", type=Path)
    parser.add_argument("--maximum-minos", required=True)
    args = parser.parse_args()
    app = args.app.expanduser().resolve(strict=True)
    maximum = _version(args.maximum_minos)
    audited = 0
    failures: list[str] = []
    observed: list[tuple[tuple[int, ...], str, Path]] = []

    for path in sorted(item for item in app.rglob("*") if item.is_file() and not item.is_symlink()):
        file_result = subprocess.run(
            ["file", "-b", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        description = file_result.stdout.strip()
        if "Mach-O" not in description:
            continue
        audited += 1
        if "arm64" not in description or "x86_64" in description:
            failures.append(f"非纯 arm64: {path.relative_to(app)} ({description})")
        build = subprocess.run(
            ["vtool", "-show-build", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        versions = re.findall(r"\bminos\s+([0-9]+(?:\.[0-9]+)+)", build.stdout)
        for value in versions:
            parsed = _version(value)
            observed.append((parsed, value, path))
            if parsed > maximum:
                failures.append(
                    f"最低系统版本 {value} 超过 {args.maximum_minos}: {path.relative_to(app)}"
                )

    if audited == 0:
        raise SystemExit("bundle 中没有找到 Mach-O 文件")
    highest = max(observed, default=((0,), "unknown", app), key=lambda item: item[0])
    print(f"mach_o_count={audited} highest_minos={highest[1]} file={highest[2].relative_to(app)}")
    if failures:
        raise SystemExit("\n".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
