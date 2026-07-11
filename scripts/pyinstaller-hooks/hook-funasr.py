"""Minimal frozen-module set for CutVideo's three bundled FunASR models."""

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("funasr", includes=["version.txt"])

# FunASR discovers model implementations through registry decorators at import
# time.  These are the dynamic modules used by fa-zh, paraformer-zh and
# fsmn-vad; their ordinary imports are followed by PyInstaller automatically.
hiddenimports = [
    "funasr.frontends.wav_frontend",
    "funasr.tokenizer.char_tokenizer",
    "funasr.models.specaug.specaug",
    "funasr.models.sanm.encoder",
    "funasr.models.monotonic_aligner.model",
    "funasr.models.bicif_paraformer.cif_predictor",
    "funasr.models.paraformer.model",
    "funasr.models.paraformer.decoder",
    "funasr.models.paraformer.cif_predictor",
    "funasr.models.fsmn_vad_streaming.model",
    "funasr.models.fsmn_vad_streaming.encoder",
]

# Registry.py calls inspect.getsourcelines() while registering classes.  Keep
# source next to the PYZ copy or FunASR silently skips the models.
module_collection_mode = {"funasr": "pyz+py"}
