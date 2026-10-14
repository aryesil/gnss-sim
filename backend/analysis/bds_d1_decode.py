"""BeiDou B1I D1 navigation message decoder (BDS-SIS-ICD-B1I-3.0).

Implements deinterleaving of BCH codewords in words 2..10, BCH(15,11,1) decoding
with error checking/correction, 300-bit subframe parsing, message assembly from
continuous symbol stream handling 180-degree carrier phase ambiguity, Keplerian
ephemeris reconstruction from subframes 1, 2, 3, and IQ demodulation with 20-bit
Neuman-Hofman secondary code wipe-off.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.analysis._bit_utils import (
    normalize_symbols,
    unpack_field,
    unpack_int,
    unpack_uint,
)
from backend.analysis._demod_utils import derotate_bpsk_carrier

PREAMBLE = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]  # 11 bits: 0x712
_BCH_G = 0b10011  # x^4 + x + 1
_PI = math.pi

# BCH(15,11,1) syndrome to flipped bit index map (0 to 14)
_SYNDROME_TO_BIT = {
    0b1001: 0,
    0b1101: 1,
    0b1111: 2,
    0b1110: 3,
    0b0111: 4,
    0b1010: 5,
    0b0101: 6,
    0b1011: 7,
    0b1100: 8,
    0b0110: 9,
    0b0011: 10,
    0b1000: 11,
    0b0100: 12,
    0b0010: 13,
    0b0001: 14,
}


def bch_syndrome(cw15: list[int] | tuple[int, ...] | np.ndarray) -> int:
    """Compute BCH(15,11,1) syndrome with g(x) = x^4 + x + 1."""
    r = 0
    for b in cw15:
        r = (r << 1) | (int(b) & 1)
    for sh in range(14, 3, -1):
        if (r >> sh) & 1:
            r ^= _BCH_G << (sh - 4)
    return r


def bch_ok(cw15: list[int] | tuple[int, ...] | np.ndarray) -> bool:
    """Check if 15-bit codeword has zero BCH syndrome (valid codeword)."""
    return bch_syndrome(cw15) == 0


def bch_decode(
    codeword_15: list[int] | tuple[int, ...] | np.ndarray,
    correct: bool = True,
) -> list[int]:
    """Decode systematic BCH(15,11,1) codeword.

    Checks parity with g(x) = x^4 + x + 1. If correct=True, corrects single-bit
    errors using syndrome lookup. Returns the 11 data bits.
    """
    cw = [int(b) & 1 for b in codeword_15]
    if len(cw) != 15:
        raise ValueError(f"expected 15 bits, got {len(cw)}")
    syn = bch_syndrome(cw)
    if syn == 0:
        return cw[:11]
    if correct and syn in _SYNDROME_TO_BIT:
        bit_idx = _SYNDROME_TO_BIT[syn]
        cw[bit_idx] ^= 1
        return cw[:11]
    return cw[:11]


def deinterleave_words(
    word_bits: list[int] | tuple[int, ...] | np.ndarray,
) -> tuple[list[int], list[int]] | list[tuple[list[int], list[int]]]:
    """Unpack bit-interleaved BCH codewords in words 2..10.

    For a 30-bit word, returns (a15, b15).
    For multiple concatenated 30-bit words (e.g. 270 bits for words 2..10),
    returns a list of (a15, b15) tuples.
    """
    wb = list(word_bits)
    if len(wb) == 30:
        return wb[0::2], wb[1::2]
    if len(wb) % 30 == 0 and len(wb) > 30:
        return [
            (wb[i : i + 30][0::2], wb[i : i + 30][1::2])
            for i in range(0, len(wb), 30)
        ]
    return wb[0::2], wb[1::2]


deinterleave2 = deinterleave_words

_u = unpack_uint
_s = unpack_int
_field = unpack_field


def decode_subframe(bits_300: list[int] | tuple[int, ...] | np.ndarray) -> dict[str, Any]:
    """Parse a 300-bit subframe, extract FraID (1..5), SOW, and 224 information bits.

    Returns dict with keys:
        'fra_id': Subframe ID (1..5)
        'subframe_id': Subframe ID (alias for fra_id)
        'sow': Seconds of week (BDT)
        'info': List of 224 information bits
        'bch_ok': True if all BCH codewords in the subframe had valid parity
    """
    b = [int(x) & 1 for x in bits_300]
    if len(b) != 300:
        raise ValueError(f"expected 300 bits, got {len(b)}")

    words = [b[i * 30 : (i + 1) * 30] for i in range(10)]
    if words[0][:11] != PREAMBLE:
        raise ValueError("preamble mismatch")

    ok = bch_ok(words[0][15:30])
    info = list(words[0][:15]) + bch_decode(words[0][15:30])

    for w in words[1:]:
        a, b_cw = deinterleave_words(w)
        ok = ok and bch_ok(a) and bch_ok(b_cw)
        info += bch_decode(a) + bch_decode(b_cw)

    fra_id = _u(info, 15, 3)
    sow = _u(info, 18, 20)
    return {
        "fra_id": fra_id,
        "subframe_id": fra_id,
        "sow": sow,
        "info": info,
        "bch_ok": bool(ok),
    }


def decode_messages(
    stream_sym01: list[int] | tuple[int, ...] | np.ndarray,
) -> list[dict[str, Any]]:
    """Search for 11-bit preamble (11100010010), handle 180-deg carrier polarity flip,
    and assemble valid subframes.
    """
    sym = normalize_symbols(stream_sym01, bpsk_sign=False)
    n = len(sym)
    for flip in (0, 1):
        stream = [x ^ flip for x in sym]
        msgs: list[dict[str, Any]] = []
        i = 0
        while i + 300 <= n:
            if stream[i : i + 11] == PREAMBLE:
                try:
                    sf = decode_subframe(stream[i : i + 300])
                    if sf["bch_ok"]:
                        msgs.append(sf)
                        i += 300
                        continue
                except (ValueError, IndexError):
                    pass
            i += 1
        if msgs:
            return msgs
    return []


def reconstruct_ephemeris(
    subframes: list[dict[str, Any]] | dict[int, Any],
    prn: int | None = None,
) -> dict[str, Any]:
    """Combine subframes 1, 2, 3 into a standard ephemeris dict for system 'C'.

    Extracts:
        system: 'C'
        bdt_week: BDT week number
        af0, af1, af2: Clock polynomial coefficients
        toe, toc: Epoch times in seconds
        tgd, tgd2: Equipment group delays
        semi-major axis (sqrtA and A), eccentricity (e)
        angles in radians (m0, omega0, omega, i0, delta_n, omega_dot, idot)
        harmonic correction terms (cuc, cus, crc, crs, cic, cis)
    """
    by_id: dict[int, list[int]] = {}
    if isinstance(subframes, dict):
        for k, v in subframes.items():
            fid = int(k)
            if isinstance(v, dict) and "info" in v:
                by_id[fid] = v["info"]
            elif isinstance(v, (list, tuple, np.ndarray)):
                by_id[fid] = [int(x) & 1 for x in v]
    elif isinstance(subframes, (list, tuple)):
        for sf in subframes:
            if isinstance(sf, dict):
                fid = sf.get("fra_id") or sf.get("subframe_id")
                if fid and "info" in sf:
                    by_id.setdefault(int(fid), sf["info"])

    if not {1, 2, 3} <= set(by_id):
        raise ValueError(f"need subframes 1, 2 and 3, got {sorted(by_id.keys())}")

    b1 = by_id[1]
    b2 = by_id[2]
    b3 = by_id[3]

    # --- Subframe 1: Clock & timing ---
    health = float(_u(b1, 38, 1))
    iodc = float(_u(b1, 39, 5))
    urai = int(_u(b1, 44, 4))
    bdt_week = int(_u(b1, 48, 13))
    toc = float(_u(b1, 61, 17) * 8.0)
    tgd = float(_s(b1, 78, 10) * 0.1e-9)
    tgd2 = float(_s(b1, 88, 10) * 0.1e-9)

    iono_alpha = tuple(
        float(_s(b1, 98 + i * 8, 8) * sc)
        for i, sc in enumerate((2**-30, 2**-27, 2**-24, 2**-24))
    )
    iono_beta = tuple(
        float(_s(b1, 130 + i * 8, 8) * sc)
        for i, sc in enumerate((2**11, 2**14, 2**16, 2**16))
    )

    af2 = float(_s(b1, 162, 11) * (2**-66))
    af0 = float(_s(b1, 173, 24) * (2**-33))
    af1 = float(_s(b1, 197, 22) * (2**-50))
    iode = float(_u(b1, 219, 5))

    # --- Subframe 2: Ephemeris 1 ---
    delta_n = float(_s(b2, 38, 16) * (2**-43) * _PI)
    cuc = float(_s(b2, 54, 18) * (2**-31))
    m0 = float(_s(b2, 72, 32) * (2**-31) * _PI)
    e = float(_u(b2, 104, 32) * (2**-33))
    cus = float(_s(b2, 136, 18) * (2**-31))
    crc = float(_s(b2, 154, 18) * (2**-6))
    crs = float(_s(b2, 172, 18) * (2**-6))
    sqrtA = float(_u(b2, 190, 32) * (2**-19))
    toe_msb = _u(b2, 222, 2)

    # --- Subframe 3: Ephemeris 2 ---
    toe_lsb = _u(b3, 38, 15)
    toe = float(((toe_msb << 15) | toe_lsb) * 8.0)
    i0 = float(_s(b3, 53, 32) * (2**-31) * _PI)
    cic = float(_s(b3, 85, 18) * (2**-31))
    omega_dot = float(_s(b3, 103, 24) * (2**-43) * _PI)
    cis = float(_s(b3, 127, 18) * (2**-31))
    idot = float(_s(b3, 145, 14) * (2**-43) * _PI)
    omega0 = float(_s(b3, 159, 32) * (2**-31) * _PI)
    omega = float(_s(b3, 191, 32) * (2**-31) * _PI)

    sow = float(_u(b1, 18, 20))

    out = {
        "system": "C",
        "bdt_week": bdt_week,
        "week": bdt_week,
        "gps_week": bdt_week + 1356,
        "toc": toc,
        "toe": toe,
        "sow": sow,
        "health": health,
        "iodc": iodc,
        "iode": iode,
        "urai": urai,
        "af0": af0,
        "af1": af1,
        "af2": af2,
        "tgd": tgd,
        "tgd2": tgd2,
        "sqrtA": sqrtA,
        "A": sqrtA**2,
        "e": e,
        "m0": m0,
        "delta_n": delta_n,
        "i0": i0,
        "idot": idot,
        "omega0": omega0,
        "omega": omega,
        "omega_dot": omega_dot,
        "cuc": cuc,
        "cus": cus,
        "crc": crc,
        "crs": crs,
        "cic": cic,
        "cis": cis,
        "iono_alpha": iono_alpha,
        "iono_beta": iono_beta,
        "iono": (iono_alpha, iono_beta),
    }
    if prn is not None:
        out["prn"] = int(prn)
    return out


# --- IQ -> B1I D1 symbols ---------------------------------------------------

_B1I_LEN = 2046
_B1I_CHIP_HZ = 2.046e6
_B1I_CTR_HZ = 1_561_098_000.0
_B1I_MS_S = 0.001
_B1I_SYM_S = 0.020  # 20 ms = 50 bps


def demod_symbols_b1i(
    iq,
    fs: float,
    prn: int,
    *,
    dopp_hz: float | None = None,
    code_phase_chips: float | None = None,
    centre_offset_hz: float = 0.0,
) -> np.ndarray:
    """Demodulate BeiDou B1I IQ into 50 bps D1 bits using B1I primary code +
    20-bit Neuman-Hofman secondary code wipe-off.

    Returns {0, 1} bits as int8 array.
    """
    from backend.analysis import l1_acquire
    from backend.synth import _lib

    iq = np.asarray(iq)
    if not np.iscomplexobj(iq):
        iq = iq.astype(np.complex64)
    elif iq.dtype == np.complex128:
        iq = iq.astype(np.complex64)

    if dopp_hz is None or code_phase_chips is None:
        acq = l1_acquire.acquire_l1(
            iq, fs, "C", prn, centre_offset_hz=centre_offset_hz
        )
        if acq.get("metric_db", -99.0) < 6.0:
            return np.zeros(0, dtype=np.int8)
        dopp_hz = acq["doppler_hz"] if dopp_hz is None else dopp_hz
        code_phase_chips = (
            acq["code_phase_chips"] if code_phase_chips is None else code_phase_chips
        )

    code, sec = _lib.code(3, int(prn), _B1I_LEN, 20)
    code = code.astype(np.float64)
    sec = sec.astype(np.float64)

    npp = int(round(fs * _B1I_MS_S))
    if npp <= 0 or len(iq) < npp * 20:
        return np.zeros(0, dtype=np.int8)

    cp0 = (_B1I_LEN - float(code_phase_chips)) % _B1I_LEN
    e0 = int(math.ceil((_B1I_LEN - cp0) % _B1I_LEN / _B1I_CHIP_HZ * fs))
    if e0 + npp * 20 > len(iq):
        e0 = 0
    base = iq[e0:]
    avail_ms = base.size // npp
    if avail_ms < 20:
        return np.zeros(0, dtype=np.int8)

    carrier_freq = centre_offset_hz + float(dopp_hz)
    rate_div_ctr = (_B1I_CHIP_HZ / _B1I_CTR_HZ) * float(dopp_hz)
    inv_fs = 1.0 / fs

    ms_corr = np.empty(avail_ms, dtype=np.complex64)
    # Bound chunk size so peak temporary buffer never exceeds ~250k samples (< 20 MB)
    max_chunk_samples = 250_000
    chunk_ms = max(20, min(100, int(max_chunk_samples // max(1, npp))))

    for m0 in range(0, avail_ms, chunk_ms):
        m1 = min(m0 + chunk_ms, avail_ms)
        n_chk = m1 - m0
        k0 = m0 * npp
        k1 = m1 * npp
        t_chk = (np.arange(k0, k1, dtype=np.float64) + e0) * inv_fs
        wipe_chk = np.exp(-2j * np.pi * carrier_freq * t_chk).astype(np.complex64)
        cph_chk = (cp0 + _B1I_CHIP_HZ * t_chk + rate_div_ctr * t_chk) % _B1I_LEN
        chip_idx_chk = np.floor(cph_chk).astype(np.int32)
        rep_chk = code[chip_idx_chk].astype(np.float32)
        ms_corr[m0:m1] = (base[k0:k1] * wipe_chk * rep_chk).reshape(n_chk, npp).sum(axis=1)

    # Search for the 20-ms secondary code alignment offset
    best_off, best_en = 0, -1.0
    for off in range(20):
        n_s = (avail_ms - off) // 20
        if n_s <= 0:
            continue
        blks = ms_corr[off : off + n_s * 20].reshape(n_s, 20)
        wiped = (blks * sec).sum(axis=1)
        en = float(np.sum(np.abs(wiped) ** 2))
        if en > best_en:
            best_off, best_en = off, en

    nsym = (avail_ms - best_off) // 20
    if nsym == 0:
        return np.zeros(0, dtype=np.int8)

    blks = ms_corr[best_off : best_off + nsym * 20].reshape(nsym, 20)
    seg = (blks * sec).sum(axis=1)
    seg = derotate_bpsk_carrier(seg, deg=2)

    return np.where(np.real(seg) > 0, np.int8(1), np.int8(0)).astype(np.int8)


demod_symbols = demod_symbols_b1i
