"""Frozen runtime files for the local-only Qwen3-ASR MLX backend."""

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files(
    "mlx_qwen3_asr",
    includes=["assets/*"],
)

# CutVideo imports several of these modules lazily so Intel/Windows builds can
# keep using FunASR.  List only the offline ASR and forced-alignment graph; the
# optional server, microphone and diarization stacks are intentionally absent.
hiddenimports = [
    "mlx_qwen3_asr",
    "mlx_qwen3_asr.attention",
    "mlx_qwen3_asr.audio",
    "mlx_qwen3_asr.cache_utils",
    "mlx_qwen3_asr.chunking",
    "mlx_qwen3_asr.config",
    "mlx_qwen3_asr.convert",
    "mlx_qwen3_asr.decoder",
    "mlx_qwen3_asr.encoder",
    "mlx_qwen3_asr.forced_aligner",
    "mlx_qwen3_asr.generate",
    "mlx_qwen3_asr.load_models",
    "mlx_qwen3_asr.model",
    "mlx_qwen3_asr.mrope",
    "mlx_qwen3_asr.runtime_utils",
    "mlx_qwen3_asr.session",
    "mlx_qwen3_asr.streaming",
    "mlx_qwen3_asr.tokenizer",
    "mlx_qwen3_asr.transcribe",
]

excludedimports = [
    "fastapi",
    "pyannote",
    "sounddevice",
    "uvicorn",
]
