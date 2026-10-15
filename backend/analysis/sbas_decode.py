"""SBAS L1 (WAAS/EGNOS/MSAS/...) navigation message decoder.

Inverse of :mod:`backend.analysis.sbas_encode`.
Implements RTCA DO-229 / ICAO Annex 10 SBAS L1:
- 250 bps data, rate-1/2 K=7 convolutional code (G1=0o171, G2=0o133, G2 not inverted)
- 250-bit block = 8-bit preamble (0x53, 0x9A, 0xC6) + 6-bit type + 212-bit data + 24-bit CRC-24Q
- Message Type 9 (GEO Navigation Message): ECEF X, Y, Z position, velocity, acceleration,
  clock corrections (tau / gamma), toe_ref, iodn, urai.
- IQ symbol demodulator (500 sym/s) for SBAS L1 signals.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from backend.analysis import _viterbi
from backend.analysis._bit_utils import (
    normalize_symbols,
    parity_bit,
    unpack_field,
    unpack_int,
    unpack_uint,
)
from backend.analysis._crc import check_crc24q, crc24q
from backend.analysis._demod_utils import derotate_bpsk_carrier

SYM_RATE_HZ = 500.0
DATA_RATE_BPS = 250.0
_PREAMBLES = (0x53, 0x9A, 0xC6)
_CRC24Q = 0x1864CFB
_G1 = _viterbi._G1
_G2 = _viterbi._G2
_NXT = _viterbi._NXT_STD
_OUT = _viterbi._OUT_STD

_POS = 0.08          # metres (DO-229 GEO position LSB)
_VEL = 0.000625      # m/s
_ACC = 0.0000050     # m/s^2

_SBAS_CHIP_HZ = 1.023e6
_SBAS_LEN = 1023
_SBAS_SYM_S = 0.002  # 2 ms = 500 sym/s
_SBAS_CTR_HZ = 1_575_420_000.0

_parity = parity_bit
_u = unpack_uint
_s = unpack_int
_field = unpack_field


def check_crc(bits250: list[int] | np.ndarray) -> bool:
    """Verify 24-bit CRC-24Q over 226 bits of header and data."""
    if len(bits250) != 250:
        return False
    return check_crc24q(bits250, data_bits=226, crc_bits=24)


def viterbi_free_decode(coded: Sequence[int] | np.ndarray) -> list[int]:
    """Exact inverse of continuous convolutional encoder on clean symbols starting from reg 0."""
    return _viterbi.viterbi_free_decode(coded, invert_g2=False)


def viterbi_decode(coded: Sequence[int] | np.ndarray) -> list[int]:
    """Hard-decision Viterbi decoder with unconstrained initial state."""
    return _viterbi.viterbi_decode(coded, invert_g2=False, known_start_state=None)


def decode_block(bits: list[int]) -> dict[str, Any] | None:
    if len(bits) != 250:
        return None
    preamble = _u(bits, 0, 8)
    mtype = _u(bits, 8, 6)
    data = list(bits[14:226])
    crc_ok = check_crc(bits)
    out: dict[str, Any] = {
        "preamble": preamble,
        "type": mtype,
        "crc_ok": crc_ok,
        "data": data,
    }
    if crc_ok:
        if mtype == 1:
            out.update(parse_type1(data))
        elif mtype == 2:
            out.update(parse_type2(data))
        elif mtype == 7:
            out.update(parse_type7(data))
        elif mtype == 9:
            out.update(parse_type9(data))
        elif mtype == 18:
            out.update(parse_type18(data))
        elif mtype == 25:
            out.update(parse_type25(data))
        elif mtype == 26:
            out.update(parse_type26(data))
    return out


def parse_type1(data_bits: list[int]) -> dict[str, Any]:
    """Parse Message Type 1 (PRN Mask) 212-bit data field."""
    mask = list(data_bits[:210])
    iodp = _u(data_bits, 210, 2)
    active_gps_prns = [i for i in range(1, 38) if mask[i - 1] == 1]
    return {
        "mask": mask,
        "iodp": iodp,
        "active_gps_prns": active_gps_prns,
    }


def parse_type2(data_bits: list[int]) -> dict[str, Any]:
    """Parse Message Type 2 (Fast Corrections) 212-bit data field."""
    iodf = _u(data_bits, 0, 2)
    iodp = _u(data_bits, 2, 2)
    prc_list = []
    udre_list = []
    for i in range(13):
        raw_prc = _s(data_bits, 4 + i * 12, 12)
        prc_m = float(raw_prc) * 0.125
        udre = _u(data_bits, 160 + i * 4, 4)
        prc_list.append(prc_m)
        udre_list.append(udre)
    return {
        "iodf": iodf,
        "iodp": iodp,
        "prc_m": prc_list,
        "udre": udre_list,
    }


def parse_type7(data_bits: list[int]) -> dict[str, Any]:
    """Parse Message Type 7 (Fast Correction Degradation Factor) 212-bit data field."""
    t_lat = _u(data_bits, 0, 4)
    iodp = _u(data_bits, 4, 2)
    ai_list = [_u(data_bits, 8 + i * 4, 4) for i in range(51)]
    return {
        "t_lat": t_lat,
        "iodp": iodp,
        "ai_list": ai_list,
    }


def parse_type18(data_bits: list[int]) -> dict[str, Any]:
    """Parse Message Type 18 (Ionospheric Grid Point Mask) 212-bit data field."""
    num_bands = _u(data_bits, 0, 4)
    band = _u(data_bits, 4, 4)
    iodi = _u(data_bits, 8, 2)
    mask = list(data_bits[10:211])
    active_igps = [idx for idx, bit in enumerate(mask) if bit == 1]
    return {
        "num_bands": num_bands,
        "band": band,
        "iodi": iodi,
        "mask": mask,
        "active_igps": active_igps,
    }


def parse_type26(data_bits: list[int]) -> dict[str, Any]:
    """Parse Message Type 26 (Ionospheric Delay Corrections) 212-bit data field."""
    band = _u(data_bits, 0, 4)
    block_id = _u(data_bits, 4, 4)
    igps = []
    for i in range(15):
        bit_off = 8 + i * 13
        raw_delay = _u(data_bits, bit_off, 9)
        givei = _u(data_bits, bit_off + 9, 4)
        delay_m = None if raw_delay == 0x1FF else float(raw_delay) * 0.125
        igps.append({"delay_m": delay_m, "givei": givei})
    iodi = _u(data_bits, 203, 2)
    return {
        "band": band,
        "block_id": block_id,
        "iodi": iodi,
        "igps": igps,
    }


def parse_type25(data_bits: list[int]) -> dict[str, Any]:
    """Parse Message Type 25 (Long-Term Satellite Error Corrections) 212-bit data field."""
    half0 = list(data_bits[:106])
    half1 = list(data_bits[106:212])

    def _parse_half(h: list[int]) -> dict[str, Any]:
        vel_code = _u(h, 0, 1)
        if vel_code == 0:
            iodp = _u(h, 103, 2)
            sats = []
            for bit_start in (1, 52):
                prn_idx = _u(h, bit_start, 6)
                if prn_idx > 0:
                    iode = _u(h, bit_start + 6, 8)
                    dx = float(_s(h, bit_start + 14, 9)) * 0.125
                    dy = float(_s(h, bit_start + 23, 9)) * 0.125
                    dz = float(_s(h, bit_start + 32, 9)) * 0.125
                    daf0 = float(_s(h, bit_start + 41, 10)) * (2.0 ** -31)
                    sats.append({
                        "prn_mask_idx": prn_idx,
                        "iode": iode,
                        "dx_m": dx,
                        "dy_m": dy,
                        "dz_m": dz,
                        "daf0_s": daf0,
                    })
            return {"vel_code": 0, "iodp": iodp, "sats": sats}
        else:
            iodp = _u(h, 104, 2)
            prn_idx = _u(h, 1, 6)
            iode = _u(h, 7, 8)
            dx = float(_s(h, 15, 11)) * 0.125
            dy = float(_s(h, 26, 11)) * 0.125
            dz = float(_s(h, 37, 11)) * 0.125
            daf0 = float(_s(h, 48, 11)) * (2.0 ** -31)
            dx_dot = float(_s(h, 59, 8)) * (2.0 ** -11)
            dy_dot = float(_s(h, 67, 8)) * (2.0 ** -11)
            dz_dot = float(_s(h, 75, 8)) * (2.0 ** -11)
            daf1 = float(_s(h, 83, 8)) * (2.0 ** -39)
            t_lt = float(_u(h, 91, 13)) * 16.0
            return {
                "vel_code": 1,
                "iodp": iodp,
                "sats": [{
                    "prn_mask_idx": prn_idx,
                    "iode": iode,
                    "dx_m": dx,
                    "dy_m": dy,
                    "dz_m": dz,
                    "daf0_s": daf0,
                    "dx_dot": dx_dot,
                    "dy_dot": dy_dot,
                    "dz_dot": dz_dot,
                    "daf1": daf1,
                    "t_lt": t_lt,
                }] if prn_idx > 0 else []
            }

    h0_res = _parse_half(half0)
    h1_res = _parse_half(half1)
    all_sats = h0_res["sats"] + h1_res["sats"]
    return {
        "halves": [h0_res, h1_res],
        "sats": all_sats,
        "iodp": h0_res["iodp"],
    }



def parse_type9(data_bits: list[int]) -> dict[str, Any]:
    """Parse Message Type 9 (GEO Navigation Message) 212-bit data field."""
    iodn = _u(data_bits, 0, 8)
    x_m = float(_s(data_bits, 8, 30)) * _POS
    y_m = float(_s(data_bits, 38, 30)) * _POS
    z_m = float(_s(data_bits, 68, 25)) * _POS
    vx = float(_s(data_bits, 93, 17)) * _VEL
    vy = float(_s(data_bits, 110, 17)) * _VEL
    vz = float(_s(data_bits, 127, 18)) * _VEL
    ax = float(_s(data_bits, 145, 10)) * _ACC
    ay = float(_s(data_bits, 155, 10)) * _ACC
    az = float(_s(data_bits, 165, 10)) * _ACC

    t0 = float(_u(data_bits, 175, 13)) * 16.0 if len(data_bits) >= 188 else 0.0
    urai = _u(data_bits, 188, 4) if len(data_bits) >= 192 else 0
    af0 = float(_s(data_bits, 192, 12)) * (2.0 ** -31) if len(data_bits) >= 204 else 0.0
    af1 = float(_s(data_bits, 204, 8)) * (2.0 ** -40) if len(data_bits) >= 212 else 0.0

    return {
        "iodn": iodn,
        "x_m": x_m,
        "y_m": y_m,
        "z_m": z_m,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "ax": ax,
        "ay": ay,
        "az": az,
        "toe_ref": t0,
        "t0": t0,
        "urai": urai,
        "af0": af0,
        "af1": af1,
        "tau": af0,
        "gamma": af1,
    }


def decode_messages(symbols: list[int] | np.ndarray) -> list[dict[str, Any]]:
    """Decode SBAS symbols ({-1, +1} or {0, 1}) into parsed message blocks.
    Handles phase inversion (180 degree ambiguity) and preamble synchronization.
    """
    b = normalize_symbols(symbols, bpsk_sign=False)

    def _find_blocks(data_bits: list[int]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        n_bits = len(data_bits)
        if n_bits < 250:
            return blocks

        for start in range(min(n_bits - 250 + 1, 500)):
            pre = _u(data_bits, start, 8)
            if pre in _PREAMBLES:
                chunk = data_bits[start : start + 250]
                if check_crc(chunk):
                    for i in range(start, n_bits - 249, 250):
                        blk_bits = data_bits[i : i + 250]
                        blk = decode_block(blk_bits)
                        if blk:
                            blocks.append(blk)
                    break
        return blocks

    def _try_decode(bit_stream: list[int]) -> list[dict[str, Any]]:
        best_blocks: list[dict[str, Any]] = []
        for sym_off in (0, 1):
            s_cand = bit_stream[sym_off:]
            data_bits = viterbi_free_decode(s_cand)
            blocks = _find_blocks(data_bits)
            if not blocks:
                data_bits = viterbi_decode(s_cand)
                blocks = _find_blocks(data_bits)
            if len([blk for blk in blocks if blk["crc_ok"]]) > len([blk for blk in best_blocks if blk["crc_ok"]]):
                best_blocks = blocks
        return best_blocks

    res = _try_decode(b)
    if not any(blk["crc_ok"] for blk in res):
        b_inv = [x ^ 1 for x in b]
        res_inv = _try_decode(b_inv)
        if len([blk for blk in res_inv if blk["crc_ok"]]) > len([blk for blk in res if blk["crc_ok"]]):
            res = res_inv

    return res


def reconstruct_ephemeris(blocks: list[dict[str, Any]], prn: int | None = None) -> dict[str, Any]:
    """Reconstruct an SBAS GEO broadcast ephemeris dict from decoded blocks."""
    t9 = None
    for blk in blocks:
        if blk.get("crc_ok") and blk.get("type") == 9:
            t9 = blk
            break
    if t9 is None:
        raise ValueError("need at least one CRC-valid SBAS Type 9 block")

    rec: dict[str, Any] = {
        "system": "S",
        "type": 9,
        "iodn": t9.get("iodn", 0),
        "x_m": t9["x_m"],
        "y_m": t9["y_m"],
        "z_m": t9["z_m"],
        "vx": t9["vx"],
        "vy": t9["vy"],
        "vz": t9["vz"],
        "ax": t9.get("ax", 0.0),
        "ay": t9.get("ay", 0.0),
        "az": t9.get("az", 0.0),
        "toe_ref": t9.get("toe_ref", 0.0),
        "tau": t9.get("tau", 0.0),
        "gamma": t9.get("gamma", 0.0),
        "af0": t9.get("af0", 0.0),
        "af1": t9.get("af1", 0.0),
        "urai": t9.get("urai", 0),
    }
    if prn is not None:
        rec["prn"] = int(prn)
    return rec


def demod_symbols_sbas_l1(
    iq,
    fs: float,
    prn: int,
    *,
    dopp_hz: float | None = None,
    code_phase_chips: float | None = None,
    centre_offset_hz: float = 0.0,
) -> np.ndarray:
    """Demodulate hard {0, 1} SBAS L1 symbols (500 sym/s) from IQ."""
    from backend.analysis import l1_acquire
    from backend.synth import _lib

    iq = np.asarray(iq)
    if not np.iscomplexobj(iq):
        iq = iq.astype(np.complex64)
    elif iq.dtype == np.complex128:
        iq = iq.astype(np.complex64)

    if dopp_hz is None or code_phase_chips is None:
        acq = l1_acquire.acquire_l1(
            iq, fs, "S", prn, centre_offset_hz=centre_offset_hz
        )
        if acq.get("metric_db", -99.0) < 6.0:
            return np.zeros(0, dtype=np.int8)
        dopp_hz = acq["doppler_hz"] if dopp_hz is None else dopp_hz
        code_phase_chips = (
            acq["code_phase_chips"] if code_phase_chips is None else code_phase_chips
        )

    code = _lib.code(2, int(prn), _SBAS_LEN, 0)[0].astype(np.float64)

    npp = int(round(fs * 0.001))
    if npp <= 0 or len(iq) < npp * 2:
        return np.zeros(0, dtype=np.int8)

    cp0 = (_SBAS_LEN - float(code_phase_chips)) % _SBAS_LEN
    e0 = int(math.ceil((_SBAS_LEN - cp0) % _SBAS_LEN / _SBAS_CHIP_HZ * fs))
    if e0 + npp * 2 > len(iq):
        e0 = 0
    base = iq[e0:]
    avail_ms = base.size // npp
    if avail_ms < 2:
        return np.zeros(0, dtype=np.int8)

    carrier_freq = centre_offset_hz + float(dopp_hz)
    rate_div_ctr = (_SBAS_CHIP_HZ / _SBAS_CTR_HZ) * float(dopp_hz)
    inv_fs = 1.0 / fs

    ms_corr = np.empty(avail_ms, dtype=np.complex64)
    # Bound chunk size so peak temporary buffer never exceeds ~250k samples (< 20 MB)
    max_chunk_samples = 250_000
    chunk_ms = max(2, min(100, int(max_chunk_samples // max(1, npp))))

    for m0 in range(0, avail_ms, chunk_ms):
        m1 = min(m0 + chunk_ms, avail_ms)
        n_chk = m1 - m0
        k0 = m0 * npp
        k1 = m1 * npp
        t_chk = (np.arange(k0, k1, dtype=np.float64) + e0) * inv_fs
        wipe_chk = np.exp(-2j * np.pi * carrier_freq * t_chk).astype(np.complex64)
        cph_chk = (cp0 + _SBAS_CHIP_HZ * t_chk + rate_div_ctr * t_chk) % _SBAS_LEN
        chip_idx_chk = np.floor(cph_chk).astype(np.int32)
        rep_chk = code[chip_idx_chk].astype(np.float32)
        ms_corr[m0:m1] = (base[k0:k1] * wipe_chk * rep_chk).reshape(n_chk, npp).sum(axis=1)

    best_off, best_en = 0, -1.0
    for off in (0, 1):
        n_s = (avail_ms - off) // 2
        if n_s <= 0:
            continue
        blks = ms_corr[off : off + n_s * 2].reshape(n_s, 2).sum(axis=1)
        en = float(np.sum(np.abs(blks) ** 2))
        if en > best_en:
            best_off, best_en = off, en

    nsym = (avail_ms - best_off) // 2
    if nsym == 0:
        return np.zeros(0, dtype=np.int8)

    seg = ms_corr[best_off : best_off + nsym * 2].reshape(nsym, 2).sum(axis=1)
    seg = derotate_bpsk_carrier(seg, deg=2)

    return np.where(np.real(seg) > 0, np.int8(1), np.int8(0)).astype(np.int8)


demod_symbols = demod_symbols_sbas_l1
