"""BeiDou B1I D1 navigation message decoder (BDS-SIS-ICD-B1I-3.0).

Inverse of backend.analysis.bds_d1_encode.
Implements BCH(15,11,1) decoding, bit deinterleaving, subframe framing,
ephemeris field extraction, Keplerian ephemeris reconstruction, and
Neuman-Hofman secondary-code wiped IQ demodulation (demod_symbols_b1i).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.analysis import bds_d1_encode as E

PREAMBLE = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]  # 11 bits
_BCH_G = 0b10011  # x^4 + x + 1
_SC = math.pi

# Neuman-Hofman 20-bit secondary code (1 kHz, bits 0 0 0 0 0 1 0 0 1 1 0 1 0 1 0 0 1 1 1 0)
# Mapped to {-1, +1} where bit 0 -> +1, bit 1 -> -1
NH20 = np.array([1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1], dtype=np.float64)


def _build_bch_tables():
    syn_to_pos = {}
    for pos in range(15):
        # error polynomial e(x) has a 1 at index pos (MSB is index 0 -> x^14)
        sh_poly = 1 << (14 - pos)
        r = sh_poly
        for sh in range(14, 3, -1):
            if (r >> sh) & 1:
                r ^= _BCH_G << (sh - 4)
        syn_to_pos[r] = pos
    return syn_to_pos


_SYN_TO_POS = _build_bch_tables()


def bch_decode(word15: list[int] | np.ndarray) -> tuple[list[int], bool]:
    """Decode systematic BCH(15,11,1) with single error correction.
    Returns (11_info_bits, is_valid_or_corrected).
    """
    w = [int(b) & 1 for b in word15]
    if len(w) != 15:
        return [0] * 11, False
    r = 0
    for b in w:
        r = (r << 1) | b
    for sh in range(14, 3, -1):
        if (r >> sh) & 1:
            r ^= _BCH_G << (sh - 4)
    if r == 0:
        return w[:11], True
    # Single bit error
    if r in _SYN_TO_POS:
        err_pos = _SYN_TO_POS[r]
        w[err_pos] ^= 1
        return w[:11], True
    return w[:11], False


def deinterleave2(word30: list[int] | np.ndarray) -> tuple[list[int], list[int]]:
    """Inverse of interleave2: unpack 30 bits into two 15-bit codewords."""
    w = list(word30)
    return w[0::2], w[1::2]


def icd_order(sf300: list[int] | np.ndarray) -> list[int]:
    """Transmitted 300 bits -> RTKLIB/ICD bit order."""
    return E.icd_order(list(sf300))


def _u(b: list[int], p: int, n: int) -> int:
    return int("".join(map(str, b[p:p + n])), 2)


def _s(b: list[int], p: int, n: int) -> int:
    v = _u(b, p, n)
    return v - (1 << n) if (v >> (n - 1)) else v


def _u2(b: list[int], p1: int, l1: int, p2: int, l2: int) -> int:
    return (_u(b, p1, l1) << l2) | _u(b, p2, l2)


def _s2(b: list[int], p1: int, l1: int, p2: int, l2: int) -> int:
    v = _u2(b, p1, l1, p2, l2)
    n = l1 + l2
    return v - (1 << n) if (v >> (n - 1)) else v


def decode_subframe(sf300: list[int] | np.ndarray) -> dict[str, Any]:
    """Decode a 300-bit BeiDou D1 subframe."""
    b = icd_order(sf300)
    if b[:11] != PREAMBLE:
        return {"bch_ok": False, "preamble_ok": False}
    words = [sf300[i * 30:(i + 1) * 30] for i in range(10)]
    w1_info, ok1 = bch_decode(words[0][15:30])
    bch_ok = ok1
    for w in words[1:]:
        a, b_w = deinterleave2(w)
        _, oka = bch_decode(a)
        _, okb = bch_decode(b_w)
        bch_ok = bch_ok and oka and okb

    fra_id = _u(b, 15, 3)
    sow = _u2(b, 18, 8, 30, 12)
    fields: dict[str, Any] = {"fra_id": fra_id, "sow": sow}

    if fra_id == 1:
        fields["week"] = _u(b, 60, 13)
        fields["toc"] = _u2(b, 73, 9, 90, 8) * 8.0
        fields["tgd"] = _s(b, 98, 10) * 0.1e-9
        fields["tgd2"] = _s2(b, 108, 4, 120, 6) * 0.1e-9
        fields["af2"] = _s(b, 214, 11) * (2.0 ** -66)
        fields["af0"] = _s2(b, 225, 7, 240, 17) * (2.0 ** -33)
        fields["af1"] = _s2(b, 257, 5, 270, 17) * (2.0 ** -50)
        fields["iode"] = _u(b, 287, 5)
        fields["iodc"] = _u(b, 43, 5)
    elif fra_id == 2:
        fields["delta_n"] = _s2(b, 42, 10, 60, 6) * (2.0 ** -43) * _SC
        fields["cuc"] = _s2(b, 66, 16, 90, 2) * (2.0 ** -31)
        fields["m0"] = _s2(b, 92, 20, 120, 12) * (2.0 ** -31) * _SC
        fields["e"] = _u2(b, 132, 10, 150, 22) * (2.0 ** -33)
        fields["cus"] = _s(b, 180, 18) * (2.0 ** -31)
        fields["crc"] = _s2(b, 198, 4, 210, 14) * (2.0 ** -6)
        fields["crs"] = _s2(b, 224, 8, 240, 10) * (2.0 ** -6)
        fields["sqrtA"] = _u2(b, 250, 12, 270, 20) * (2.0 ** -19)
        fields["toe_msb"] = _u(b, 290, 2)
    elif fra_id == 3:
        fields["toe_lsb"] = _u2(b, 42, 10, 60, 5)
        fields["i0"] = _s2(b, 65, 17, 90, 15) * (2.0 ** -31) * _SC
        fields["cic"] = _s2(b, 105, 7, 120, 11) * (2.0 ** -31)
        fields["omega_dot"] = _s2(b, 131, 11, 150, 13) * (2.0 ** -43) * _SC
        fields["cis"] = _s2(b, 163, 9, 180, 9) * (2.0 ** -31)
        fields["idot"] = _s2(b, 189, 13, 210, 1) * (2.0 ** -43) * _SC
        fields["omega0"] = _s2(b, 211, 21, 240, 11) * (2.0 ** -31) * _SC
        fields["omega"] = _s2(b, 251, 11, 270, 21) * (2.0 ** -31) * _SC

    return {
        "fra_id": fra_id,
        "sow": sow,
        "bch_ok": bch_ok,
        "preamble_ok": True,
        "fields": fields,
        "icd_bits": b,
    }


def decode_messages(sym01: list[int] | np.ndarray) -> list[dict[str, Any]]:
    """Scan a {0, 1} symbol stream (50 bps) for BeiDou D1 subframes (300 bits).
    Handles preamble search, frame alignment, and 180-degree polarity ambiguity.
    """
    sym = [int(x) & 1 for x in sym01]
    n = len(sym)
    pre = PREAMBLE

    for flip in (0, 1):
        stream = [x ^ flip for x in sym]
        msgs = []
        i = 0
        while i + 300 <= n:
            if stream[i:i + 11] == pre:
                sf = decode_subframe(stream[i:i + 300])
                if sf.get("bch_ok"):
                    msgs.append(sf)
                    i += 300
                    continue
            i += 1
        if msgs:
            return msgs
    return []


def reconstruct_ephemeris(msgs: list[dict[str, Any]], *, prn: int = 1) -> dict[str, Any]:
    """Reconstruct a complete BeiDou ephemeris record from subframes 1, 2, and 3."""
    by_fraid = {}
    for m in msgs:
        if m.get("bch_ok") and m.get("fields"):
            fraid = m["fra_id"]
            by_fraid[fraid] = m["fields"]

    required = {1, 2, 3}
    if not required.issubset(by_fraid):
        raise ValueError(f"need BCH-valid BeiDou D1 subframes 1, 2 and 3, got {sorted(by_fraid.keys())}")

    f1 = by_fraid[1]
    f2 = by_fraid[2]
    f3 = by_fraid[3]

    toe = float(((f2["toe_msb"] << 15) | f3["toe_lsb"]) * 8.0)

    rec = {
        "system": "C",
        "prn": prn,
        "bdt_week": int(f1["week"]),
        "toc": float(f1["toc"]),
        "tgd": float(f1["tgd"]),
        "tgd2": float(f1.get("tgd2", 0.0)),
        "af0": float(f1["af0"]),
        "af1": float(f1["af1"]),
        "af2": float(f1["af2"]),
        "iode": int(f1["iode"]),
        "iodc": int(f1.get("iodc", 0)),
        "delta_n": float(f2["delta_n"]),
        "cuc": float(f2["cuc"]),
        "m0": float(f2["m0"]),
        "e": float(f2["e"]),
        "cus": float(f2["cus"]),
        "crc": float(f2["crc"]),
        "crs": float(f2["crs"]),
        "sqrtA": float(f2["sqrtA"]),
        "toe": toe,
        "i0": float(f3["i0"]),
        "cic": float(f3["cic"]),
        "omega_dot": float(f3["omega_dot"]),
        "cis": float(f3["cis"]),
        "idot": float(f3["idot"]),
        "omega0": float(f3["omega0"]),
        "omega": float(f3["omega"]),
    }
    return rec


_B1I_LEN = 2046
_B1I_CHIP_HZ = 2.046e6
_B1I_CTR_HZ = 1_561_098_000.0  # config.B1I_HZ
_B1I_MS_S = 0.001              # 1 ms period = 2046 chips


def demod_symbols_b1i(
    iq,
    fs: float,
    prn: int,
    *,
    dopp_hz: float | None = None,
    code_phase_chips: float | None = None,
    centre_offset_hz: float = 0.0,
    is_geo: bool = False,
) -> np.ndarray:
    """Demodulate hard {0, 1} D1 navigation bits (50 bps) from BeiDou B1I IQ.
    Wipes Doppler and correlates against B1I primary code at 1 kHz,
    syncs/wipes the 20-bit Neuman-Hofman secondary code, and resolves BPSK bits.
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

    code = _lib.code(3, int(prn), _B1I_LEN, 0)[0].astype(np.float64)

    npp = int(round(fs * _B1I_MS_S))  # samples per 1 ms period
    nms = len(iq) // npp
    if nms < 20:
        return np.zeros(0, dtype=np.int8)

    cp0 = (_B1I_LEN - float(code_phase_chips)) % _B1I_LEN
    e0 = int(math.ceil((_B1I_LEN - cp0) % _B1I_LEN / _B1I_CHIP_HZ * fs))
    if e0 + npp * 20 > len(iq):
        e0 = 0
    base = iq[e0:]
    nms = base.size // npp
    if nms < 20:
        return np.zeros(0, dtype=np.int8)

    t = (np.arange(nms * npp, dtype=np.float64) + e0) / fs
    carrier_freq = centre_offset_hz + float(dopp_hz)
    wipe = np.exp(-2j * np.pi * carrier_freq * t)

    cph = (
        cp0 + _B1I_CHIP_HZ * t + (_B1I_CHIP_HZ / _B1I_CTR_HZ) * float(dopp_hz) * t
    ) % _B1I_LEN
    chip_idx = np.floor(cph).astype(np.int64)
    rep = code[chip_idx]

    # 1 ms correlation values (1000 sps)
    corr_1ms = (base[: nms * npp] * wipe * rep).reshape(nms, npp).sum(axis=1)

    if is_geo:
        # Flat secondary code (+1)
        nbits = nms // 20
        seg = corr_1ms[: nbits * 20].reshape(nbits, 20).sum(axis=1)
    else:
        # MEO/IGSO: 20-bit Neuman-Hofman secondary code
        # Find NH20 alignment offset (0..19)
        nh = NH20
        best_off = 0
        best_score = -1.0
        for off in range(20):
            avail = (nms - off) // 20
            if avail < 1:
                continue
            blocks = corr_1ms[off : off + avail * 20].reshape(avail, 20)
            score = float(np.sum(np.abs(np.sum(blocks * nh, axis=1))))
            if score > best_score:
                best_score = score
                best_off = off

        nbits = (nms - best_off) // 20
        if nbits == 0:
            return np.zeros(0, dtype=np.int8)
        blocks = corr_1ms[best_off : best_off + nbits * 20].reshape(nbits, 20)
        seg = np.sum(blocks * nh, axis=1)

    # Carrier phase drift removal
    ph = np.unwrap(np.angle(seg ** 2))
    deg = 2 if nbits >= 8 else 1
    bi = np.arange(nbits, dtype=np.float64)
    coef = np.polyfit(bi, ph, deg) / 2.0
    seg = seg * np.exp(-1j * np.polyval(coef, bi))
    phi0 = 0.5 * np.angle(np.sum(seg ** 2))
    seg = seg * np.exp(-1j * phi0)

    # Note: bit 0 -> +1 (real > 0 -> 0), bit 1 -> -1 (real < 0 -> 1)
    return np.where(np.real(seg) > 0, np.int8(0), np.int8(1)).astype(np.int8)


demod_symbols = demod_symbols_b1i
