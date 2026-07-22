# Third-party notices

This application is distributed with unmodified third-party runtime components. The corresponding
license texts shipped by the installed wheels are preserved under
`resources/licenses/python-packages/`; model and FFmpeg license texts are under
`resources/licenses/`. `resources/manifest.json` records the pinned model and binary hashes.

## Qt for Python / PySide6 6.11.1

Qt for Python is available under LGPLv3/GPL and commercial terms. This build uses the LGPLv3 option
for unmodified dynamically linked Qt libraries. The complete standalone application directory is
distributed so the Qt libraries remain separate files. LGPLv3/GPLv3 and wheel notices are included.

## NumPy 2.4.6

NumPy is distributed under the BSD 3-Clause license and includes separately licensed components.
The wheel's complete `licenses/` tree is included.

## pypinyin 0.55.0

pypinyin is distributed under the MIT license. It is used entirely offline to match common Chinese
ASR homophone substitutions without sending transcript or audio data to any external service. The
license text supplied by the wheel is included.

## PyTorch 2.13.0 and torchaudio 2.11.0

PyTorch and torchaudio use BSD-style licenses and contain third-party components. They are retained
for the Windows/FunASR distribution. The Apple Silicon Qwen/MLX package excludes PyTorch and
torchaudio from the frozen application.

## FunASR 1.3.14, ModelScope 1.38.1 and speech models

FunASR code is MIT-licensed. ModelScope code is Apache-2.0 licensed. The three bundled model
snapshots (`fa-zh`, `paraformer-zh`, `fsmn-vad`) each declare Apache-2.0; their source IDs,
snapshot revisions and tree hashes are recorded in the manifest. A model's license is separate
from the FunASR repository license. These models are retained for the Windows distribution and are
removed from the Apple Silicon Qwen/MLX package.

## Qwen3-ASR 0.6B, Qwen3 Forced Aligner 0.6B and mlx-qwen3-asr 0.3.5

Apple Silicon builds use local 4-bit MLX conversions of `Qwen/Qwen3-ASR-0.6B` and
`Qwen/Qwen3-ForcedAligner-0.6B`. The upstream model cards declare Apache-2.0. The converted model
tree hashes, upstream URLs and quantization format are recorded in the resource manifest.
`mlx-qwen3-asr` is an Apache-2.0 implementation of Qwen3-ASR for Apple's MLX runtime; its wheel
license is preserved under `resources/licenses/python-packages/`.

## MLX 0.29.4, MLX Metal 0.29.4 and regex 2026.7.19

MLX and MLX Metal are distributed under the MIT license. Their native libraries and Metal shader
bundle remain separate runtime files inside the application. The Python `regex` package retains
the license text shipped by its wheel. Complete wheel metadata and license files for these macOS
runtime components are included under `resources/licenses/python-packages/`.

## FFmpeg

FFmpeg is LGPLv2.1-or-later by default, but optional GPL/nonfree components change its obligations.
The bundled FFmpeg builds must not enable GPL/nonfree components. The release includes the complete
`ffmpeg -buildconf` output, exact binary hashes and source URL. The Apple Silicon build may link the
LGPL libmp3lame library; that library is kept beside the executable with its license.

This notice is operational guidance and attribution, not legal advice.
