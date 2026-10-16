"""Backend acquisition for any L1-group GNSS system (GPS, QZSS, SBAS, Galileo,
BeiDou, GLONASS). GLONASS FDMA frequency offsets are applied internally via
glo_k, and Galileo E1 replicas include the BOC(1,1) subcarrier modulation.
"""
from __future__ import annotations

import numpy as np

from backend.synth import _lib
from backend.synth.engine import _glo_g1_code
from backend.synth.signals import SIGNALS

_SIGNAL_KEYS = {
    "G": "GPS_L1CA",
    "J": "QZSS_L1CA",
    "S": "SBAS_L1",
    "E": "GAL_E1",
    "C": "BDS_B1I",
    "R": "GLO_G1",
}

_CODEGEN_ENUM = {
    "G": 0,
    "J": 1,
    "S": 2,
    "C": 3,
    "E": 6,
}


def acquire_l1(
    iq,
    fs: float,
    sysc: str,
    prn: int,
    *,
    centre_offset_hz: float,
    glo_k: int | None = None,
    dopp_hz: float = 5000.0,
    dopp_step_hz: float = 200.0,
    nperiods: int = 4,
) -> dict:
    if sysc not in ("G", "J", "S", "E", "C", "R"):
        raise ValueError(f"unsupported system {sysc!r}")
    if sysc == "R":
        # GLONASS kanalları: -7..+6 ve +7 teknik kanalı kapsar
        if glo_k is None or not isinstance(glo_k, int) or isinstance(glo_k, bool) or glo_k not in range(-7, 8):
            raise ValueError("GLONASS acquisition needs glo_k in [-7, 7]")

    sig = SIGNALS[_SIGNAL_KEYS[sysc]]
    code_len = sig.code_len
    chip_hz = sig.chip_rate_hz

    if sysc == "R":
        code = _glo_g1_code().astype(np.float64)
    else:
        enum = _CODEGEN_ENUM[sysc]
        sec_len = 25 if sysc == "E" else 0
        code = _lib.code(enum, prn, code_len, sec_len)[0].astype(np.float64)

    carrier_center = centre_offset_hz + (glo_k * 562_500.0 if sysc == "R" else 0.0)

    period_s = code_len / chip_hz
    npp = int(round(fs * period_s))
    avail = len(iq) // npp
    if avail < 1:
        return {
            "metric_db": -99.0,
            "doppler_hz": 0.0,
            "code_phase_chips": 0.0,
            "code_len": int(code_len),
            "chip_hz": float(chip_hz),
        }

    nc = min(nperiods, avail)
    seg = np.asarray(iq[: npp * nc], dtype=np.complex128)

    t = np.arange(npp) / fs
    chip_phase = t * chip_hz
    idx = np.floor(chip_phase).astype(int) % code_len
    local = code[idx].astype(np.float64)
    if sysc == "E":
        sub = np.where(np.floor(2.0 * chip_phase).astype(np.int64) & 1, -1.0, 1.0)
        local = local * sub
    LOC = np.conj(np.fft.fft(local))

    dopps = np.arange(-dopp_hz, dopp_hz + 1, dopp_step_hz)
    best_pk = -1.0
    best = (0.0, 0)
    best_floor = 1.0
    for fd in dopps:
        acc = np.zeros(npp)
        for k in range(nc):
            blk = seg[k * npp : (k + 1) * npp]
            ph = np.exp(
                -1j * 2 * np.pi * (carrier_center + fd) * (np.arange(k * npp, (k + 1) * npp)) / fs
            )
            acc += np.abs(np.fft.ifft(np.fft.fft(blk * ph) * LOC)) ** 2
        pk = acc.max()
        if pk > best_pk:
            best_pk = pk
            best = (float(fd), int(acc.argmax()))
            best_floor = float(np.median(acc))

    return {
        "metric_db": float(10 * np.log10(best_pk / max(best_floor, 1e-9))),
        "doppler_hz": best[0],
        "code_phase_chips": float((best[1] * chip_hz / fs) % code_len),
        "code_len": int(code_len),
        "chip_hz": float(chip_hz),
    }
