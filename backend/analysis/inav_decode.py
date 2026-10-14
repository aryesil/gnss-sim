"""Galileo E1 I/NAV navigation message decoder.

The inverse of backend.analysis.inav_encode.
Implements deinterleaving, rate-1/2 convolutional decoding, page framing,
CRC-24Q verification, word type 0-5 extraction, and Keplerian ephemeris
reconstruction from a CRC-valid set of words 1, 2, 3, 4.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.analysis import _inav_layout as L
from backend.analysis._crc import crc24q

_G1 = 0o171
_G2 = 0o133
_PI = math.pi


def _parity(x: int) -> int:
    return bin(x).count("1") & 1


def deinterleave_30x8(sym) -> list[int]:
    """Block deinterleaver: read 8 rows x 30 cols that were written by rows and read by cols."""
    a = np.asarray(list(sym)).reshape(30, 8)
    return a.T.reshape(L.CODED_SYMS).tolist()


def viterbi_free_decode(coded) -> list[int]:
    """Exact inverse of inav_encode.conv_encode for clean channel from initial state 0."""
    reg = 0
    out: list[int] = []
    coded = [int(s) & 1 for s in coded]
    for i in range(0, len(coded) - 1, 2):
        c1, c2 = coded[i], coded[i + 1]
        for cand in (0, 1):
            r = ((reg << 1) | cand) & 0x7F
            if _parity(r & _G1) == c1 and (_parity(r & _G2) ^ 1) == c2:
                out.append(cand)
                reg = r
                break
        else:
            out.append(0)
            reg = (reg << 1) & 0x7F
    return out


def _viterbi_tables():
    """Trellis transition tables for rate-1/2 K=7 code with G2 inverted."""
    nxt = np.zeros((64, 2), dtype=np.int64)
    out = np.zeros((64, 2, 2), dtype=np.int64)
    for st in range(64):
        for b in (0, 1):
            r = ((st << 1) | b) & 0x7F
            nxt[st, b] = r & 0x3F
            out[st, b, 0] = _parity(r & _G1)
            out[st, b, 1] = _parity(r & _G2) ^ 1
    return nxt, out


_NXT, _OUT = _viterbi_tables()


def viterbi_decode(coded) -> list[int]:
    """Hard-decision Viterbi decoder for rate-1/2 K=7 convolutional code with G2 inverted.
    Known starting state is 0 (each page part starts from state 0).
    """
    s = np.asarray([int(x) & 1 for x in coded], dtype=np.int64)
    n = s.size // 2
    if n == 0:
        return []
    pairs = s[: 2 * n].reshape(n, 2)
    metric = np.full(64, np.inf)
    metric[0] = 0.0

    prev = np.zeros((n, 64), dtype=np.int64)
    bit = np.zeros((n, 64), dtype=np.int8)
    src = np.repeat(np.arange(64), 2)
    bsrc = np.tile([0, 1], 64)
    dst = _NXT[src, bsrc]

    for k in range(n):
        cost = metric[src] + (_OUT[src, bsrc, 0] != pairs[k, 0]) + (_OUT[src, bsrc, 1] != pairs[k, 1])
        order = np.lexsort((cost, dst))
        first = np.ones(order.size, dtype=bool)
        first[1:] = dst[order][1:] != dst[order][:-1]
        win = order[first]
        new = np.full(64, np.inf)
        new[dst[win]] = cost[win]
        prev[k, dst[win]] = src[win]
        bit[k, dst[win]] = bsrc[win]
        metric = new

    st = int(np.argmin(metric))
    bits = np.zeros(n, dtype=np.int8)
    for k in range(n - 1, -1, -1):
        bits[k] = bit[k, st]
        st = int(prev[k, st])
    return bits.tolist()


def _u(bits: list[int], lo: int = 0, n: int | None = None) -> int:
    if n is None:
        n = len(bits) - lo
    v = 0
    for b in bits[lo : lo + n]:
        v = (v << 1) | (int(b) & 1)
    return v


def _field(bits: list[int], lo: int, kind: str, scale: float, n: int) -> float | int:
    v = _u(bits, lo, n)
    if kind == "s" and (v & (1 << (n - 1))):
        v -= 1 << n
    return v * scale


def decode_word_fields(wtype: int, word_bits: list[int]) -> dict[str, Any]:
    """Decode 128-bit word into named fields using the layout table."""
    out: dict[str, Any] = {}
    if wtype not in L.LAYOUT:
        return out
    off = 0
    for name, kind, scale, n in L.LAYOUT[wtype]:
        if not name.startswith("_"):
            out[name] = _field(word_bits, off, kind, scale, n)
        off += n
    return out


def decode_page(even_syms: list[int], odd_syms: list[int]) -> dict[str, Any] | None:
    """Decode a 500-symbol nominal page from even and odd 250-symbol page parts."""
    if len(even_syms) != L.HALF_PAGE_SYMS or len(odd_syms) != L.HALF_PAGE_SYMS:
        return None

    # Deinterleave 240 symbols
    even_coded = deinterleave_30x8(even_syms[L.SYNC_SYMS:])
    odd_coded = deinterleave_30x8(odd_syms[L.SYNC_SYMS:])

    # Decode convolutional code (try free decode first, fallback to Viterbi)
    even_bits = viterbi_free_decode(even_coded)
    odd_bits = viterbi_free_decode(odd_coded)

    # Validate even/odd bit and page type:
    # even: even/odd bit == 0, page type == 0
    # odd:  even/odd bit == 1, page type == 0
    if len(even_bits) < 120 or len(odd_bits) < 120:
        return None
    if even_bits[0] != 0 or odd_bits[0] != 1:
        # Try full Viterbi decode
        even_bits = viterbi_decode(even_coded)
        odd_bits = viterbi_decode(odd_coded)
        if even_bits[0] != 0 or odd_bits[0] != 1:
            return None

    data_k = even_bits[2:114]
    data_j = odd_bits[2:18]
    reserved_1 = odd_bits[18:58]
    sar = odd_bits[58:80]
    spare = odd_bits[80:82]
    crc_rx = odd_bits[82:106]

    crc_in = [0, 0] + data_k + [1, 0] + data_j + reserved_1 + sar + spare
    crc_calc = crc24q(crc_in)
    crc_ok = (crc_calc == crc_rx)

    word_bits = data_k + data_j
    wtype = _u(word_bits, 0, 6)

    return {
        "type": wtype,
        "crc_ok": crc_ok,
        "word_bits": word_bits,
        "fields": decode_word_fields(wtype, word_bits) if crc_ok else {},
    }


def decode_messages(sym01: list[int] | np.ndarray) -> list[dict[str, Any]]:
    """Scan a {0, 1} symbol stream for Galileo E1 I/NAV pages.
    Handles sync pattern searching, page part pairing, CRC-24Q checking,
    and carrier phase ambiguity (polarity flip).
    """
    sym = [int(x) & 1 for x in sym01]
    n = len(sym)
    sync = L.SYNC_PATTERN

    for flip in (0, 1):
        stream = [x ^ flip for x in sym]
        msgs: list[dict[str, Any]] = []
        i = 0
        while i + L.PAGE_SYMS <= n:
            # Check even part sync
            if stream[i : i + L.SYNC_SYMS] == sync:
                # Check odd part sync at i + 250
                if stream[i + L.HALF_PAGE_SYMS : i + L.HALF_PAGE_SYMS + L.SYNC_SYMS] == sync:
                    even_part = stream[i : i + L.HALF_PAGE_SYMS]
                    odd_part = stream[i + L.HALF_PAGE_SYMS : i + L.PAGE_SYMS]
                    page = decode_page(even_part, odd_part)
                    if page is not None and page["crc_ok"]:
                        msgs.append(page)
                        i += L.PAGE_SYMS
                        continue
            i += 1
        if msgs:
            return msgs
    return []


def reconstruct_ephemeris(msgs: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct Galileo Keplerian ephemeris dictionary from CRC-valid I/NAV messages.
    Requires word types 1, 2, 3, 4. Word types 5 and 0 are used when available for
    GST week, TOW, and group delays.
    """
    by_type: dict[int, dict[str, Any]] = {}
    for m in msgs:
        if m.get("crc_ok") and m.get("fields"):
            wtype = m["type"]
            by_type[wtype] = m["fields"]

    required = {1, 2, 3, 4}
    if not required.issubset(by_type):
        raise ValueError(f"need CRC-valid I/NAV word types 1, 2, 3 and 4, got {sorted(by_type.keys())}")

    f1 = by_type[1]
    f2 = by_type[2]
    f3 = by_type[3]
    f4 = by_type[4]
    f5 = by_type.get(5, {})
    f0 = by_type.get(0, {})

    rec: dict[str, Any] = {"system": "E"}
    rec["prn"] = int(f4["prn"])
    rec["iodnav"] = int(f1["iodnav"])

    # Orbital parameters
    rec["sqrtA"] = float(f1["sqrtA"])
    rec["e"] = float(f1["e"])
    rec["m0"] = float(f1["m0"]) * _PI
    rec["toe"] = float(f1["toe"])

    rec["omega0"] = float(f2["omega0"]) * _PI
    rec["i0"] = float(f2["i0"]) * _PI
    rec["omega"] = float(f2["omega"]) * _PI
    rec["idot"] = float(f2["idot"]) * _PI

    rec["omega_dot"] = float(f3["omega_dot"]) * _PI
    rec["delta_n"] = float(f3["delta_n"]) * _PI
    rec["cuc"] = float(f3["cuc"])
    rec["cus"] = float(f3["cus"])
    rec["crc"] = float(f3["crc"])
    rec["crs"] = float(f3["crs"])
    if "sisa" in f3:
        rec["sisa"] = float(f3["sisa"])

    # Harmonic corrections and clock
    rec["cic"] = float(f4["cic"])
    rec["cis"] = float(f4["cis"])
    rec["toc"] = float(f4["toc"])
    rec["af0"] = float(f4["af0"])
    rec["af1"] = float(f4["af1"])
    rec["af2"] = float(f4["af2"])

    # Group delays and timing from word 5 / word 0
    if f5:
        rec["tgd"] = float(f5.get("bgd_e1_e5b", 0.0))
        rec["tgd_e5a"] = float(f5.get("bgd_e1_e5a", rec["tgd"]))
        rec["gal_week"] = int(f5.get("week", 0))
        rec["tow"] = float(f5.get("tow", 0.0))
    elif f0:
        rec["gal_week"] = int(f0.get("week", 0))
        rec["tow"] = float(f0.get("tow", 0.0))
        rec["tgd"] = 0.0
        rec["tgd_e5a"] = 0.0
    else:
        rec["gal_week"] = 0
        rec["tow"] = 0.0
        rec["tgd"] = 0.0
        rec["tgd_e5a"] = 0.0

    return rec


# --- IQ -> I/NAV symbols -----------------------------------------------------

_E1_LEN = 4092
_E1_CHIP_HZ = 1.023e6
_E1_CTR_HZ = 1_575_420_000.0
_E1_SYM_S = 0.004  # 4 ms = 250 sym/s


def demod_symbols_e1b(
    iq,
    fs: float,
    prn: int,
    *,
    dopp_hz: float | None = None,
    code_phase_chips: float | None = None,
    centre_offset_hz: float = 0.0,
) -> np.ndarray:
    """Demodulate hard {0, 1} I/NAV symbols (250 sym/s) from Galileo E1 IQ.

    Wipes carrier Doppler and correlates against the Galileo E1-B primary code
    modulated by the BOC(1,1) subcarrier (1.023 MHz).
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
            iq, fs, "E", prn, centre_offset_hz=centre_offset_hz
        )
        if acq.get("metric_db", -99.0) < 6.0:
            return np.zeros(0, dtype=np.int8)
        dopp_hz = acq["doppler_hz"] if dopp_hz is None else dopp_hz
        code_phase_chips = (
            acq["code_phase_chips"] if code_phase_chips is None else code_phase_chips
        )

    code = _lib.code(5, int(prn), _E1_LEN, 0)[0].astype(np.float64)

    npp = int(round(fs * _E1_SYM_S))
    nsym = len(iq) // npp
    if nsym == 0:
        return np.zeros(0, dtype=np.int8)

    cp0 = (_E1_LEN - float(code_phase_chips)) % _E1_LEN
    e0 = int(math.ceil((_E1_LEN - cp0) % _E1_LEN / _E1_CHIP_HZ * fs))
    if e0 + npp * 4 > len(iq):
        e0 = 0
    base = iq[e0:]
    nsym = base.size // npp
    if nsym == 0:
        return np.zeros(0, dtype=np.int8)

    t = (np.arange(nsym * npp, dtype=np.float64) + e0) / fs
    carrier_freq = centre_offset_hz + float(dopp_hz)
    wipe = np.exp(-2j * np.pi * carrier_freq * t)

    cph = (
        cp0 + _E1_CHIP_HZ * t + (_E1_CHIP_HZ / _E1_CTR_HZ) * float(dopp_hz) * t
    ) % _E1_LEN
    chip_idx = np.floor(cph).astype(np.int64)
    sub = np.where(np.floor(2.0 * cph).astype(np.int64) & 1, -1.0, 1.0)
    rep = code[chip_idx] * sub

    seg = (base[: nsym * npp] * wipe * rep).reshape(nsym, npp).sum(axis=1)

    ph = np.unwrap(np.angle(seg ** 2))
    deg = 2 if nsym >= 8 else 1
    bi = np.arange(nsym, dtype=np.float64)
    coef = np.polyfit(bi, ph, deg) / 2.0
    seg = seg * np.exp(-1j * np.polyval(coef, bi))
    phi0 = 0.5 * np.angle(np.sum(seg ** 2))
    seg = seg * np.exp(-1j * phi0)

    return np.where(np.real(seg) > 0, np.int8(1), np.int8(0)).astype(np.int8)


demod_symbols = demod_symbols_e1b

