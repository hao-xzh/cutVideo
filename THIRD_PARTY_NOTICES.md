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

PyTorch and torchaudio use BSD-style licenses and contain third-party components. The complete
license trees supplied by their CPU wheels are included. The application does not include CUDA
models or require a GPU.

## FunASR 1.3.14, ModelScope 1.38.1 and speech models

FunASR code is MIT-licensed. ModelScope code is Apache-2.0 licensed. The three bundled model
snapshots (`fa-zh`, `paraformer-zh`, `fsmn-vad`) each declare Apache-2.0; their source IDs,
snapshot revisions and tree hashes are recorded in the manifest. A model's license is separate
from the FunASR repository license.

## FFmpeg

FFmpeg is LGPLv2.1-or-later by default, but optional GPL/nonfree components change its obligations.
The bundled FFmpeg builds must not enable GPL/nonfree components. The release includes the complete
`ffmpeg -buildconf` output, exact binary hashes and source URL. The Apple Silicon build may link the
LGPL libmp3lame library; that library is kept beside the executable with its license.

This notice is operational guidance and attribution, not legal advice.
