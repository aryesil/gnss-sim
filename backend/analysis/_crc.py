"""CRC-24Q (polynomial 0x1864CFB) calculation and verification.

Parity checksum used by GPS CNAV, Galileo I/NAV and F/NAV, SBAS L1, and RTCM.
Direct form CRC-24Q: MSB-first over data bits (M(x)*x^24 mod g(x)).
Accelerated with precomputed 256-entry byte lookup table.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np

_CRC24Q = 0x1864CFB


def _build_crc24q_table() -> list[int]:
    table: list[int] = []
    for byte in range(256):
        reg = byte << 16
        for _ in range(8):
            if reg & 0x800000:
                reg = ((reg << 1) ^ _CRC24Q) & 0xFFFFFF
            else:
                reg = (reg << 1) & 0xFFFFFF
        table.append(reg)
    return table


_CRC24Q_TABLE = _build_crc24q_table()


def crc24q(bits: Sequence[int] | np.ndarray) -> list[int]:
    """Compute 24-bit CRC-24Q checksum for a bit sequence using fast byte-table lookup.

    Parameters
    ----------
    bits : Sequence[int] | np.ndarray
        Payload bits (0s and 1s).

    Returns
    -------
    list[int]
        24-bit CRC-24Q checksum as a list of binary integers.
    """
    reg = 0
    n = len(bits)
    idx = 0

    # Fast 8-bit table processing for full bytes
    while idx + 8 <= n:
        chunk = bits[idx : idx + 8]
        byte = (
            ((int(chunk[0]) & 1) << 7)
            | ((int(chunk[1]) & 1) << 6)
            | ((int(chunk[2]) & 1) << 5)
            | ((int(chunk[3]) & 1) << 4)
            | ((int(chunk[4]) & 1) << 3)
            | ((int(chunk[5]) & 1) << 2)
            | ((int(chunk[6]) & 1) << 1)
            | (int(chunk[7]) & 1)
        )
        top = (reg >> 16) ^ byte
        reg = ((reg << 8) & 0xFFFFFF) ^ _CRC24Q_TABLE[top]
        idx += 8

    # Process remaining 0-7 residual bits
    for b in bits[idx:]:
        reg ^= (int(b) & 1) << 23
        reg <<= 1
        if reg & (1 << 24):
            reg ^= _CRC24Q
        reg &= 0xFFFFFF

    # Doğrudan form CRC-24Q zaten M(x)*x^24 mod g(x) hesaplamıştır;
    # arkadan 24 sıfır kaydırmak standardı bozar, bu yüzden trailing shift uygulanmaz.
    return [(reg >> (23 - i)) & 1 for i in range(24)]


def check_crc24q(
    bits: Sequence[int] | np.ndarray,
    data_bits: int,
    crc_bits: int = 24,
) -> bool:
    """Verify trailing CRC-24Q checksum against message payload.

    Parameters
    ----------
    bits : Sequence[int] | np.ndarray
        Full sequence containing data_bits followed by crc_bits checksum.
    data_bits : int
        Number of payload bits preceding the CRC.
    crc_bits : int, default 24
        Number of trailing CRC bits (standard is 24).

    Returns
    -------
    bool
        True if computed CRC matches the trailing checksum exactly; False otherwise.
    """
    total = data_bits + crc_bits
    if len(bits) < total:
        return False
    payload = bits[:data_bits]
    expected = [int(b) & 1 for b in bits[data_bits:total]]
    return crc24q(payload) == expected
