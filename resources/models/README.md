# Bundled model layout

Release builds load these directories by absolute local path and set all supported offline flags.
They never pass a registry alias to FunASR at runtime.

- `fa-zh/`: `iic/speech_timestamp_prediction-v1-16k-offline`
- `paraformer-zh/`: `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- `fsmn-vad/`: `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`

Model weights are intentionally not committed to GitHub. Use
`scripts/fetch_models.py --accept-model-licenses` during release preparation; the script downloads
the exact commits recorded in `resources/manifest.json`, and packaging refuses a tree-hash mismatch.
