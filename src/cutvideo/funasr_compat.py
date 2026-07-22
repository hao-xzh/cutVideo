"""Narrow runtime compatibility fixes for the bundled FunASR release."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

_LOGGER = logging.getLogger(__name__)
_PATCH_LOCK = threading.Lock()
_TIMESTAMP_PATCH_MARKER = "_cutvideo_timestamp_mismatch_compat"
_MPS_CIF_PATCH_MARKER = "_cutvideo_mps_cif_compat"


def _install_mps_cif_compatibility() -> None:
    """Keep Paraformer's CIF predictor on MPS without losing prefix precision.

    FunASR calculates one small CIF prefix-sum tensor in float64 and converts
    the result back to float32.  Apple MPS cannot create float64 tensors, so an
    otherwise MPS-compatible Paraformer graph fails during its first forward
    pass.  For MPS only, calculate that small prefix sum on CPU in the same
    float64 precision and move the float32 result back to the original device.
    The encoder, decoder and remaining CIF work continue to run on the GPU.

    This is intentionally narrower than replacing the operation with an MPS
    float32 cumsum: CutVideo depends on CIF boundaries for editable timestamps,
    so preserving FunASR's existing accumulation precision is preferable to a
    tiny additional CPU/GPU transfer per VAD segment.
    """

    from funasr.models.paraformer import cif_predictor  # type: ignore[import-not-found]

    current: Callable[..., Any] = cif_predictor.cif_wo_hidden_v1
    if getattr(current, _MPS_CIF_PATCH_MARKER, False):
        return

    @wraps(current)
    def cif_wo_hidden_v1_mps_compat(
        alphas: Any,
        threshold: Any,
        return_fire_idxs: bool = False,
    ) -> Any:
        if getattr(getattr(alphas, "device", None), "type", "") != "mps":
            return current(alphas, threshold, return_fire_idxs)

        torch = cif_predictor.torch
        batch_size, len_time = alphas.size()
        device = alphas.device
        dtype = alphas.dtype
        fires = torch.zeros(batch_size, len_time, dtype=dtype, device=device)

        # FunASR immediately converts its float64 cumsum back to float32.  Do
        # exactly the same calculation, with only that unsupported operation
        # on CPU.  Inference does not need an autograd edge across this helper.
        prefix_sum = torch.cumsum(
            # Transfer before casting: a combined ``to(cpu, float64)`` asks
            # MPS to perform the unsupported float64 conversion itself.
            alphas.detach().to(device="cpu").to(dtype=torch.float64),
            dim=1,
        ).to(device=device, dtype=torch.float32)
        prefix_sum_floor = torch.floor(prefix_sum)
        dislocation_prefix_sum = torch.roll(prefix_sum, 1, dims=1)
        dislocation_prefix_sum_floor = torch.floor(dislocation_prefix_sum)
        dislocation_prefix_sum_floor[:, 0] = 0
        dislocation_diff = prefix_sum_floor - dislocation_prefix_sum_floor
        fire_idxs = dislocation_diff > 0
        fires[fire_idxs] = 1
        fires = fires + prefix_sum - prefix_sum_floor
        if return_fire_idxs:
            return fires, fire_idxs
        return fires

    setattr(cif_wo_hidden_v1_mps_compat, _MPS_CIF_PATCH_MARKER, True)
    cif_predictor.cif_wo_hidden_v1 = cif_wo_hidden_v1_mps_compat


def install_funasr_compatibility() -> None:
    """Keep FunASR text when its timestamp predictor returns fewer entries.

    FunASR 1.3.14's ``sentence_postprocess`` indexes timestamps using the
    decoded-token index.  On a real mixed-language segment the timestamp
    predictor can legitimately return fewer entries than the decoder, causing
    an ``IndexError`` and aborting the complete long-form transcription.

    The fallback deliberately keeps only timestamps the model actually
    returned.  CutVideo's result adapter then groups the full recognized text
    over those real intervals and marks the precision as token-level instead
    of inventing character timestamps.
    """

    from funasr.utils import postprocess_utils  # type: ignore[import-not-found]

    with _PATCH_LOCK:
        _install_mps_cif_compatibility()
        current: Callable[..., Any] = postprocess_utils.sentence_postprocess
        if getattr(current, _TIMESTAMP_PATCH_MARKER, False):
            return

        @wraps(current)
        def sentence_postprocess_compat(
            words: list[Any],
            time_stamp: Sequence[Sequence[Any]] | None = None,
        ) -> Any:
            if time_stamp is None:
                return current(words, None)
            try:
                return current(words, time_stamp)
            except IndexError:
                sentence, real_word_lists = current(words, None)
                recovered_timestamps = list(time_stamp)
                _LOGGER.warning(
                    "FunASR timestamp count mismatch; preserving %d real timestamps "
                    "for %d decoded tokens",
                    len(recovered_timestamps),
                    len(words),
                )
                return sentence, recovered_timestamps, real_word_lists

        setattr(sentence_postprocess_compat, _TIMESTAMP_PATCH_MARKER, True)
        postprocess_utils.sentence_postprocess = sentence_postprocess_compat


__all__ = ["install_funasr_compatibility"]
