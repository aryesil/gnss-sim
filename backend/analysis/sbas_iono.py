"""RTCA DO-229 / ICAO Annex 10 SBAS Message Type 18 & 26 (Ionosphere) generator.

Generates DO-229 Message Type 18 (Ionospheric Grid Point Mask) and Message
Type 26 (Ionospheric Delay Corrections) from real IONEX or synthetic VTEC maps.
Maintains DO-229 continuous rate-1/2 convolutional FEC at 500 sym/s.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Sequence, Any

import numpy as np

from backend.analysis._crc import crc24q
from backend.analysis import sbas_encode

SYM_RATE_HZ = 500.0
_PREAMBLES = (0x53, 0x9A, 0xC6)

# DO-229 Table A-14 Ionospheric Mask Bands definitions
# Bands 0 to 8: Mercator bands (lon_step = 5 deg, 8 meridian columns per band)
# Bands 9 and 10: Polar bands (lat_step = 5 or 10 deg)
_X1 = [-75, -65, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5,
       0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 65, 75, 85]
_X2 = [-55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5,
       0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
_X3 = [-75, -65, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5,
       0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 65, 75]
_X4 = [-85, -75, -65, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5,
       0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 65, 75]
_X5 = list(range(-180, 180, 5))
_X6 = list(range(-180, 180, 10))
_X7 = list(range(-180, 180, 30))
_X8 = [-170, -140, -110, -80, -50, -20, 10, 40, 70, 100, 130, 160]

# Band tables: band -> list of (lon, lats, bit_start, bit_end)
IGP_BANDS: dict[int, list[tuple[int, list[int], int, int]]] = {
    0: [(-180, _X1, 1, 28), (-175, _X2, 29, 51), (-170, _X3, 52, 78), (-165, _X2, 79, 101),
        (-160, _X3, 102, 128), (-155, _X2, 129, 151), (-150, _X3, 152, 178), (-145, _X2, 179, 201)],
    1: [(-140, _X4, 1, 28), (-135, _X2, 29, 51), (-130, _X3, 52, 78), (-125, _X2, 79, 101),
        (-120, _X3, 102, 128), (-115, _X2, 129, 151), (-110, _X3, 152, 178), (-105, _X2, 179, 201)],
    2: [(-100, _X3, 1, 27), (-95, _X2, 28, 50), (-90, _X1, 51, 78), (-85, _X2, 79, 101),
        (-80, _X3, 102, 128), (-75, _X2, 129, 151), (-70, _X3, 152, 178), (-65, _X2, 179, 201)],
    3: [(-60, _X3, 1, 27), (-55, _X2, 28, 50), (-50, _X4, 51, 78), (-45, _X2, 79, 101),
        (-40, _X3, 102, 128), (-35, _X2, 129, 151), (-30, _X3, 152, 178), (-25, _X2, 179, 201)],
    4: [(-20, _X3, 1, 27), (-15, _X2, 28, 50), (-10, _X3, 51, 77), (-5, _X2, 78, 100),
        (0, _X1, 101, 128), (5, _X2, 129, 151), (10, _X3, 152, 178), (15, _X2, 179, 201)],
    5: [(20, _X3, 1, 27), (25, _X2, 28, 50), (30, _X3, 51, 77), (35, _X2, 78, 100),
        (40, _X4, 101, 128), (45, _X2, 129, 151), (50, _X3, 152, 178), (55, _X2, 179, 201)],
    6: [(60, _X3, 1, 27), (65, _X2, 28, 50), (70, _X3, 51, 77), (75, _X2, 78, 100),
        (80, _X3, 101, 127), (85, _X2, 128, 150), (90, _X1, 151, 178), (95, _X2, 179, 201)],
    7: [(100, _X3, 1, 27), (105, _X2, 28, 50), (110, _X3, 51, 77), (115, _X2, 78, 100),
        (120, _X3, 101, 127), (125, _X2, 128, 150), (130, _X4, 151, 178), (135, _X2, 179, 201)],
    8: [(140, _X3, 1, 27), (145, _X2, 28, 50), (150, _X3, 51, 77), (155, _X2, 78, 100),
        (160, _X3, 101, 127), (165, _X2, 128, 150), (170, _X3, 151, 177), (175, _X2, 178, 200)],
    9: [(60, _X5, 1, 72), (65, _X6, 73, 108), (70, _X6, 109, 144), (75, _X6, 145, 180),
        (85, _X7, 181, 192)],
    10: [(-60, _X5, 1, 72), (-65, _X6, 73, 108), (-70, _X6, 109, 144), (-75, _X6, 145, 180),
         (-85, _X8, 181, 192)],
}


def get_band_igps(band: int) -> list[tuple[float, float]]:
    """Return 1-indexed list of (lat, lon) coordinates for the given band."""
    cols = IGP_BANDS.get(band, [])
    igps: list[tuple[float, float]] = []
    for lon, lats, b_start, b_end in cols:
        for lat in lats:
            if band <= 8:
                igps.append((float(lat), float(lon)))
            else:
                # Polar bands: first field is lat, lats list is actually lons
                igps.append((float(lon), float(lat)))
    return igps


def find_best_band(lat_deg: float, lon_deg: float) -> int:
    """Find the DO-229 band covering (lat_deg, lon_deg)."""
    if lat_deg > 60.0:
        return 9
    if lat_deg < -60.0:
        return 10
    lon = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    # Bands 0..8 span 40-degree longitudinal sectors:
    # Band 0: [-180, -140), Band 1: [-140, -100), ... Band 4: [-20, 20), Band 5: [20, 60)
    b = int(math.floor((lon + 180.0) / 40.0))
    return max(0, min(8, b))


def find_bands_for_region(lat_deg: float, lon_deg: float, lon_window_deg: float = 25.0) -> list[int]:
    """RTCA DO-229 Bölüm A.4.4.10 uyarınca alıcının görünür bölgesini kapsayan tüm iyonosfer bandlarını belirler."""
    bands = set()
    if lat_deg > 55.0:
        bands.add(9)
    if lat_deg < -55.0:
        bands.add(10)

    lon_center = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    for offset in np.linspace(-lon_window_deg, lon_window_deg, 11):
        sample_lon = ((lon_center + offset + 180.0) % 360.0) - 180.0
        b = int(math.floor((sample_lon + 180.0) / 40.0))
        b = max(0, min(8, b))
        bands.add(b)

    return sorted(bands)


def _bits_of(value: int, n: int) -> list[int]:
    v = int(value) & ((1 << n) - 1)
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def encode_type18(band: int, mask_bits: Sequence[int], iodi: int = 0,
                  num_bands: int = 1) -> list[int]:
    """Build 212-bit data field for Message Type 18 (Ionospheric Grid Point Mask).
    - num_bands: 4 bits
    - band: 4 bits
    - iodi: 2 bits
    - mask: 201 bits
    - spare: 1 bit
    Total: 212 bits.
    """
    b = []
    b.extend(_bits_of(num_bands, 4))
    b.extend(_bits_of(band, 4))
    b.extend(_bits_of(iodi, 2))
    m = list(mask_bits[:201])
    if len(m) < 201:
        m.extend([0] * (201 - len(m)))
    b.extend(m[:201])
    b.append(0)  # spare bit
    return b[:212]


def build_type18_block(idx: int, band: int, mask_bits: Sequence[int],
                       iodi: int = 0, num_bands: int = 1) -> list[int]:
    """Build full 250-bit Message Type 18 block with preamble and CRC-24Q."""
    pre = _bits_of(_PREAMBLES[idx % 3], 8)
    data = encode_type18(band, mask_bits, iodi=iodi, num_bands=num_bands)
    head = pre + _bits_of(18, 6) + data
    return head + sbas_encode.crc24q(head)


def encode_type1(gps_prns: Sequence[int], iodp: int = 0) -> list[int]:
    """Build 212-bit data field for Message Type 1 (PRN Mask).
    - mask: 210 bits (bits 1..37 map to GPS PRN 1..37)
    - iodp: 2 bits
    Total: 212 bits.
    """
    mask = [0] * 210
    for p in gps_prns:
        p_int = int(p)
        if 1 <= p_int <= 37:
            mask[p_int - 1] = 1
    return mask + _bits_of(iodp, 2)


def build_type1_block(idx: int, gps_prns: Sequence[int], iodp: int = 0) -> list[int]:
    """Build full 250-bit Message Type 1 block with preamble and CRC-24Q."""
    pre = _bits_of(_PREAMBLES[idx % 3], 8)
    data = encode_type1(gps_prns, iodp=iodp)
    head = pre + _bits_of(1, 6) + data
    return head + sbas_encode.crc24q(head)


def encode_type2(prc_list: Sequence[float] | None = None,
                 udre_list: Sequence[int] | None = None,
                 iodf: int = 0, iodp: int = 0) -> list[int]:
    """Build 212-bit data field for Message Type 2 (Fast Corrections).
    - iodf: 2 bits
    - iodp: 2 bits (matches MT 1)
    - 13 PRCs (13 * 12 = 156 bits, signed, scale 0.125 m)
    - 13 UDREs (13 * 4 = 52 bits, e.g. 3 for sigma <= 1.75m)
    Total: 212 bits.
    """
    b = []
    b.extend(_bits_of(iodf, 2))
    b.extend(_bits_of(iodp, 2))

    prcs = list(prc_list) if prc_list is not None else []
    udres = list(udre_list) if udre_list is not None else []

    for i in range(13):
        if i < len(prcs):
            val = prcs[i]
            q = int(round(val / 0.125))
            q = max(-2047, min(2047, q))
            b.extend(sbas_encode.twos(q * 0.125, 0.125, 12))
        else:
            b.extend([0] * 12)  # nominal 0.0 m correction

    for i in range(13):
        if i < len(udres):
            u = int(udres[i])
            b.extend(_bits_of(max(0, min(15, u)), 4))
        else:
            b.extend(_bits_of(3, 4))  # standard healthy UDRE (sigma <= 1.75 m)

    return b[:212]


def build_type2_block(idx: int, prc_list: Sequence[float] | None = None,
                      udre_list: Sequence[int] | None = None,
                      iodf: int = 0, iodp: int = 0) -> list[int]:
    """Build full 250-bit Message Type 2 block with preamble and CRC-24Q."""
    pre = _bits_of(_PREAMBLES[idx % 3], 8)
    data = encode_type2(prc_list, udre_list, iodf=iodf, iodp=iodp)
    head = pre + _bits_of(2, 6) + data
    return head + sbas_encode.crc24q(head)


def encode_type7(ai_list: Sequence[int] | None = None,
                 t_lat: int = 0, iodp: int = 0) -> list[int]:
    """Build 212-bit data field for Message Type 7 (Fast Correction Degradation Factor).
    - t_lat: 4 bits (System Latency, 0..15 s)
    - iodp: 2 bits (matches MT 1 IODP)
    - spare: 2 bits (0)
    - 51 degradation factor indicators (51 * 4 = 204 bits, ai = 1 for nominal active SVs)
    Total: 212 bits.
    """
    b = []
    b.extend(_bits_of(t_lat, 4))
    b.extend(_bits_of(iodp, 2))
    b.extend([0, 0])  # spare 2 bits

    ais = list(ai_list) if ai_list is not None else []
    for i in range(51):
        if i < len(ais):
            val = max(0, min(15, int(ais[i])))
        else:
            val = 0
        b.extend(_bits_of(val, 4))

    return b[:212]


def build_type7_block(idx: int, ai_list: Sequence[int] | None = None,
                      t_lat: int = 0, iodp: int = 0) -> list[int]:
    """Build full 250-bit Message Type 7 block with preamble and CRC-24Q."""
    pre = _bits_of(_PREAMBLES[idx % 3], 8)
    data = encode_type7(ai_list, t_lat=t_lat, iodp=iodp)
    head = pre + _bits_of(7, 6) + data
    return head + sbas_encode.crc24q(head)



def encode_type26(band: int, block_id: int, delays_m: Sequence[float],
                  givei_list: Sequence[int], iodi: int = 0) -> list[int]:
    """Build 212-bit data field for Message Type 26 (Ionospheric Delay Corrections).
    - band: 4 bits
    - block_id: 4 bits (0..13)
    - 15 IGPs (15 * 13 = 195 bits):
        - delay (9 bits, scale 0.125 m, 0x1FF = 511 if unmonitored)
        - givei (4 bits, 0..15, 15 if unmonitored)
    - iodi: 2 bits
    - spare: 7 bits
    Total: 212 bits.
    """
    b = []
    b.extend(_bits_of(band, 4))
    b.extend(_bits_of(block_id, 4))

    for i in range(15):
        if i < len(delays_m):
            d_m = delays_m[i]
            q_d = int(round(d_m / 0.125))
            q_d = max(0, min(510, q_d))
            gv = int(givei_list[i]) if i < len(givei_list) else 3
            gv = max(0, min(15, gv))
        else:
            q_d = 511  # unmonitored / unavailable
            gv = 15
        b.extend(_bits_of(q_d, 9))
        b.extend(_bits_of(gv, 4))

    b.extend(_bits_of(iodi, 2))
    b.extend([0] * 7)  # spare
    return b[:212]


def build_type26_block(idx: int, band: int, block_id: int,
                       delays_m: Sequence[float], givei_list: Sequence[int],
                       iodi: int = 0) -> list[int]:
    """Build full 250-bit Message Type 26 block with preamble and CRC-24Q."""
    pre = _bits_of(_PREAMBLES[idx % 3], 8)
    data = encode_type26(band, block_id, delays_m, givei_list, iodi=iodi)
    head = pre + _bits_of(26, 6) + data
    return head + sbas_encode.crc24q(head)


def encode_type25(sat_corrections: Sequence[tuple[int, int, float, float, float, float] | dict[str, Any]],
                  iodp: int = 0) -> list[int]:
    """Build 212-bit data field for Message Type 25 (Long-Term Satellite Error Corrections, Velocity Code = 0).
    Can encode up to 4 satellites (2 in first half, 2 in second half).
    Each satellite entry can be a tuple:
      (prn_mask_idx, iode, dx_m, dy_m, dz_m, daf0_s)
    or a dict with keys ('prn_mask_idx', 'iode', 'dx_m', 'dy_m', 'dz_m', 'daf0_s').
    """
    entries = []
    for sc in sat_corrections[:4]:
        if isinstance(sc, dict):
            entries.append((
                int(sc.get("prn_mask_idx", 0)),
                int(sc.get("iode", 0)),
                float(sc.get("dx_m", 0.0)),
                float(sc.get("dy_m", 0.0)),
                float(sc.get("dz_m", 0.0)),
                float(sc.get("daf0_s", 0.0)),
            ))
        else:
            p_idx, iode_val = int(sc[0]), int(sc[1])
            dx = float(sc[2]) if len(sc) > 2 else 0.0
            dy = float(sc[3]) if len(sc) > 3 else 0.0
            dz = float(sc[4]) if len(sc) > 4 else 0.0
            daf0 = float(sc[5]) if len(sc) > 5 else 0.0
            entries.append((p_idx, iode_val, dx, dy, dz, daf0))

    def _encode_sat_slot(entry: tuple[int, int, float, float, float, float] | None) -> list[int]:
        if entry is None or entry[0] <= 0:
            return [0] * 51
        p_idx, iode_val, dx, dy, dz, daf0 = entry
        b = []
        b.extend(_bits_of(p_idx, 6))
        b.extend(_bits_of(iode_val & 0xFF, 8))
        for d in (dx, dy, dz):
            q = int(round(d / 0.125))
            q = max(-256, min(255, q))
            b.extend(sbas_encode.twos(q * 0.125, 0.125, 9))
        q_daf0 = int(round(daf0 / (2.0 ** -31)))
        q_daf0 = max(-512, min(511, q_daf0))
        b.extend(sbas_encode.twos(q_daf0 * (2.0 ** -31), 2.0 ** -31, 10))
        return b[:51]

    # Half 0 (first 2 satellites)
    s0 = entries[0] if len(entries) > 0 else None
    s1 = entries[1] if len(entries) > 1 else None
    h0 = [0]  # vel code = 0
    h0.extend(_encode_sat_slot(s0))
    h0.extend(_encode_sat_slot(s1))
    h0.extend(_bits_of(iodp, 2))
    h0.append(0)  # spare bit
    # total: 1 + 51 + 51 + 2 + 1 = 106 bits

    # Half 1 (next 2 satellites)
    s2 = entries[2] if len(entries) > 2 else None
    s3 = entries[3] if len(entries) > 3 else None
    h1 = [0]  # vel code = 0
    h1.extend(_encode_sat_slot(s2))
    h1.extend(_encode_sat_slot(s3))
    h1.extend(_bits_of(iodp, 2))
    h1.append(0)  # spare bit
    # total: 1 + 51 + 51 + 2 + 1 = 106 bits

    return (h0 + h1)[:212]


def build_type25_block(idx: int,
                       sat_corrections: Sequence[tuple[int, int, float, float, float, float] | dict[str, Any]],
                       iodp: int = 0) -> list[int]:
    """Build full 250-bit Message Type 25 block with preamble and CRC-24Q."""
    pre = _bits_of(_PREAMBLES[idx % 3], 8)
    data = encode_type25(sat_corrections, iodp=iodp)
    head = pre + _bits_of(25, 6) + data
    return head + sbas_encode.crc24q(head)
 
 
def encode_type9(eph: dict[str, Any], tow_s: float = 0.0, iodn: int = 0, urai: int = 0) -> list[int]:
    """Build 212-bit data field for Message Type 9 (GEO Navigation Message)
    strictly compliant with RTCA DO-229 Table A-8 / ICAO Annex 10.

    Fields:
    - IODN: 8 bits (0..255)
    - t_0,GEO: 13 bits (Time of day count in 16 s units, 0..5399)
    - URAI: 4 bits (0 = healthy, sigma <= 2.4 m; 15 = unhealthy / do not use)
    - X_G: 30 bits, scale 0.08 m
    - Y_G: 30 bits, scale 0.08 m
    - Z_G: 25 bits, scale 0.4 m
    - X_dot_G: 17 bits, scale 0.000625 m/s
    - Y_dot_G: 17 bits, scale 0.000625 m/s
    - Z_dot_G: 18 bits, scale 0.004 m/s
    - X_ddot_G: 10 bits, scale 0.0000125 m/s^2
    - Y_ddot_G: 10 bits, scale 0.0000125 m/s^2
    - Z_ddot_G: 10 bits, scale 0.0000625 m/s^2
    - a_Gf0: 12 bits, scale 2^-31 s
    - a_Gf1: 8 bits, scale 2^-40 s/s
    Total: exactly 212 bits.
    """
    b = []
    # 1. IODN: 8 bits
    i_val = int(eph.get("iodn", iodn) or 0) & 0xFF
    b.extend(_bits_of(i_val, 8))

    # 2. t_0,GEO: 13 bits (0..5399)
    t0_s = eph.get("t0", eph.get("toe_ref", tow_s))
    try:
        t0_val = (int(float(t0_s)) % 86400) // 16
    except Exception:
        t0_val = 0
    t0_val = max(0, min(5399, t0_val))
    b.extend(_bits_of(t0_val, 13))

    # 3. URAI: 4 bits (0 = healthy, sigma <= 2.4 m; 15 = unhealthy / do not use)
    u_val = int(eph.get("urai", urai) or 0) & 0x0F
    b.extend(_bits_of(u_val, 4))

    # 4. Position: X (30b, 0.08m), Y (30b, 0.08m), Z (25b, 0.4m)
    # georinex ve ephemeris parser değerleri doğrudan SI (metre, m/s, m/s^2) cinsinden sağlar.
    x_m = float(eph.get("x_m", 0.0) or 0.0)
    y_m = float(eph.get("y_m", 0.0) or 0.0)
    z_m = float(eph.get("z_m", 0.0) or 0.0)
    b.extend(sbas_encode.twos(x_m, 0.08, 30))
    b.extend(sbas_encode.twos(y_m, 0.08, 30))
    b.extend(sbas_encode.twos(z_m, 0.4, 25))

    # 5. Velocity: VX (17b, 0.000625 m/s), VY (17b, 0.000625 m/s), VZ (18b, 0.004 m/s)
    vx = float(eph.get("vx", 0.0) or 0.0)
    vy = float(eph.get("vy", 0.0) or 0.0)
    vz = float(eph.get("vz", 0.0) or 0.0)
    b.extend(sbas_encode.twos(vx, 0.000625, 17))
    b.extend(sbas_encode.twos(vy, 0.000625, 17))
    b.extend(sbas_encode.twos(vz, 0.004, 18))

    # 6. Acceleration: AX (10b, 0.0000125 m/s^2), AY (10b, 0.0000125 m/s^2), AZ (10b, 0.0000625 m/s^2)
    b.extend(sbas_encode.twos(eph.get("ax", 0.0), 0.0000125, 10))
    b.extend(sbas_encode.twos(eph.get("ay", 0.0), 0.0000125, 10))
    b.extend(sbas_encode.twos(eph.get("az", 0.0), 0.0000625, 10))

    # 7. Clock: af0 (12b, 2^-31 s), af1 (8b, 2^-40 s/s)
    af0_s = eph.get("af0", eph.get("tau", 0.0))
    af1_s = eph.get("af1", eph.get("gamma", 0.0))
    b.extend(sbas_encode.twos(af0_s, 2.0 ** -31, 12))
    b.extend(sbas_encode.twos(af1_s, 2.0 ** -40, 8))

    return b[:212]


def build_type9_block(idx: int, eph: dict[str, Any], tow_s: float = 0.0) -> list[int]:
    """Build full 250-bit Message Type 9 block with preamble and CRC-24Q."""
    pre = _bits_of(_PREAMBLES[idx % 3], 8)
    data = encode_type9(eph, tow_s=tow_s)
    head = pre + _bits_of(9, 6) + data
    return head + sbas_encode.crc24q(head)



def select_active_igps(band: int, rx_lat: float, rx_lon: float,
                       radius_deg: float = 25.0) -> tuple[list[int], list[int]]:
    """Select active IGPs in band within radius_deg of rx_lat, rx_lon.
    Returns (mask_201_bits, active_indices_0_based).
    """
    all_igps = get_band_igps(band)
    mask = [0] * 201
    active_indices = []

    for idx, (lat, lon) in enumerate(all_igps):
        d_lat = lat - rx_lat
        # Longitude difference handling wrap
        d_lon = ((lon - rx_lon + 180.0) % 360.0) - 180.0
        dist = math.hypot(d_lat, d_lon * math.cos(math.radians(rx_lat)))
        if dist <= radius_deg:
            mask[idx] = 1
            active_indices.append(idx)

    # Ensure at least the closest grid points are active if none fell in radius
    if not active_indices and all_igps:
        dists = [
            math.hypot(lat - rx_lat, (((lon - rx_lon + 180.0) % 360.0) - 180.0) * math.cos(math.radians(rx_lat)))
            for lat, lon in all_igps
        ]
        best_idx = int(np.argmin(dists))
        mask[best_idx] = 1
        active_indices.append(best_idx)

    return mask, active_indices


def nav_stream(eph: dict, week: int, tow0_sow: float, duration_s: float,
               eph_by_prn=None, ionex_map=None,
               rx_lat: float | None = None, rx_lon: float | None = None,
               gps_prns: Sequence[int] | None = None) -> tuple[np.ndarray, float]:
    """Deterministic SBAS L1 symbol stream with Type 9 (GEO) + Type 1 (PRN Mask) +
    Type 2 (Fast Corrections) + Type 18 (Mask) + Type 26 (Iono Delays).
    Continuous rate-1/2 FEC, 500 sym/s (int8 in {-1, +1}).
    """
    from backend.ephem import ionex

    imap = ionex_map
    if imap is None:
        # Simülasyon tarihini week ve tow0_sow'dan türet; week <= 0 veya yoksa
        # testlerde ağ erişimi yapılmaması için sentetik haritaya düş.
        if week is not None and week > 0:
            target_dt = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc) + dt.timedelta(weeks=week, seconds=float(tow0_sow))
            imap = ionex.load_or_fetch_ionex(target_dt)
        else:
            imap = ionex.synthetic_ionex()

    # Determine reference receiver location: if not given, use GEO sub-satellite point
    if rx_lat is None or rx_lon is None:
        x_m = float(eph.get("x_m", 0.0))
        y_m = float(eph.get("y_m", 0.0))
        geo_lon = math.degrees(math.atan2(y_m, x_m)) if (x_m != 0.0 or y_m != 0.0) else 0.0
        ref_lat = 40.0
        ref_lon = geo_lon
    else:
        ref_lat = float(rx_lat)
        ref_lon = float(rx_lon)

    # Determine GPS satellites to augment with MT 1 and MT 2
    active_gps: list[int] = []
    if gps_prns:
        active_gps = sorted(set(int(p) for p in gps_prns if 1 <= int(p) <= 32))
    elif eph_by_prn and isinstance(eph_by_prn, dict):
        for k in eph_by_prn.keys():
            if isinstance(k, tuple) and len(k) >= 2 and k[0] == "G":
                active_gps.append(int(k[1]))
            elif isinstance(k, int) and 1 <= k <= 32:
                active_gps.append(k)
        active_gps = sorted(set(active_gps))
    if not active_gps:
        active_gps = list(range(1, 14))  # default 13 PRNs to populate MT 2 cleanly
    active_gps = active_gps[:13]          # MT 2 handles first 13 PRNs

    # RTCA DO-229 Bölüm A.4.4.10 uyarınca alıcının görünür bölgesini kapsayan tüm bandları belirle:
    bands = find_bands_for_region(ref_lat, ref_lon, lon_window_deg=25.0)
    num_bands = len(bands)

    band_masks = {}
    blocks_26 = []
    for b in bands:
        all_b_igps = get_band_igps(b)
        b_mask_bits, b_active_indices = select_active_igps(b, ref_lat, ref_lon, radius_deg=18.0)
        band_masks[b] = b_mask_bits
        delays_b = [
            imap.get_vertical_delay_m(all_b_igps[idx][0], all_b_igps[idx][1], tow0_sow)
            for idx in b_active_indices
        ]
        gives_b = [3] * len(b_active_indices)
        n_b_blocks = max(1, int(math.ceil(len(b_active_indices) / 15.0)))
        for b_id in range(n_b_blocks):
            b_delays = delays_b[b_id * 15 : (b_id + 1) * 15]
            b_gives = gives_b[b_id * 15 : (b_id + 1) * 15]
            blocks_26.append((b, b_id, b_delays, b_gives))

    # Build DO-229 Message Type 25 (Long-Term Corrections) with matching IODE for each GPS SV
    sat_corrections = []
    for prn_idx_0, p in enumerate(active_gps):
        mask_idx = prn_idx_0 + 1
        p_rec = None
        if isinstance(eph_by_prn, dict):
            p_rec = eph_by_prn.get(p) or eph_by_prn.get(("G", p)) or eph_by_prn.get(str(p))
        iode = int(p_rec.get("iode", 0) or 0) & 0xFF if isinstance(p_rec, dict) else 0
        sat_corrections.append((mask_idx, iode, 0.0, 0.0, 0.0, 0.0))

    n_mt25_blocks = max(1, int(math.ceil(len(sat_corrections) / 4.0)))
    blocks_25 = []
    for b_id in range(n_mt25_blocks):
        blocks_25.append(sat_corrections[b_id * 4 : (b_id + 1) * 4])

    # Pre-build blocks for the schedule
    iodp = 0
    iodi = 0
    blk_mt1 = lambda k: build_type1_block(k, active_gps, iodp=iodp)
    blk_mt2 = lambda k: build_type2_block(k, prc_list=[0.0] * len(active_gps),
                                          udre_list=[3] * len(active_gps),
                                          iodf=0, iodp=iodp)
    ai_list = [1] * len(active_gps)
    blk_mt7 = lambda k: build_type7_block(k, ai_list, t_lat=0, iodp=iodp)

    nblk = int(math.ceil((math.ceil(duration_s) + 30)))
    conv = sbas_encode._Conv()
    syms: list[int] = []

    k0 = int(math.floor(float(tow0_sow)))

    # DO-229 10-second repeating cycle:
    # 0: Type 9 (GEO Ephemeris & Clock)
    # 1: Type 1 (PRN Mask)
    # 2: Type 2 (Fast Corrections)
    # 3: Type 18 (IGP Mask)
    # 4: Type 26 (Iono Delays - cycling block)
    # 5: Type 25 (Long-Term Corrections - Block 0, 2, ...)
    # 6: Type 2 (Fast Corrections)
    # 7: Type 7 (Fast Correction Degradation Factor)
    # 8: Type 25 (Long-Term Corrections - Block 1, 3, ...)
    # 9: Type 2 (Fast Corrections)
    # Segment yeniden başlatmada (canlı mod) konvolüsyonel kodlayıcı yazmacının
    # sürekliliğini sağlamak için döngü k0-1 bloğundan başlatılır (warm-up),
    # ardından ilk 500 sembol atılarak gerçek yayındaki yazmaç durumu korunur.
    for i in range(-1, nblk):
        k = k0 + i
        sec = k % 10
        cycle = k // 10
        if sec == 0:
            blk = build_type9_block(k, eph, tow_s=float(k))
        elif sec == 1:
            blk = blk_mt1(k)
        elif sec in (2, 6, 9):
            blk = blk_mt2(k)
        elif sec == 3:
            # Type 18 (IGP Mask): Görünür tüm bandlar cycle % num_bands ile periyodik olarak yayınlanır.
            b18_band = bands[cycle % num_bands]
            blk = build_type18_block(k, b18_band, band_masks[b18_band], iodi=iodi, num_bands=num_bands)
        elif sec == 4:
            # Type 26 (Iono Delays): Tüm bandların blokları sıra ile döndürülür.
            b26_idx = cycle % len(blocks_26)
            b_band, b_id, b_delays, b_gives = blocks_26[b26_idx]
            blk = build_type26_block(k, b_band, b_id, b_delays, b_gives, iodi=iodi)
        elif sec == 5:
            b25_idx = (cycle * 2) % len(blocks_25)
            blk = build_type25_block(k, blocks_25[b25_idx], iodp=iodp)
        elif sec == 7:
            blk = blk_mt7(k)
        elif sec == 8:
            b25_idx = (cycle * 2 + 1) % len(blocks_25)
            blk = build_type25_block(k, blocks_25[b25_idx], iodp=iodp)
        else:
            blk = sbas_encode.build_block(k, 63, eph)

        syms.extend(conv.encode(blk))

    syms = syms[500:]
    arr = np.asarray(syms, dtype=np.int8)
    return np.where(arr > 0, np.int8(1), np.int8(-1)).astype(np.int8), SYM_RATE_HZ
