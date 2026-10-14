"""Shared bit packing, unpacking, field decoding, parity, and symbol utilities.

Centralizes low-level bit-level operations across GPS, Galileo, BeiDou, and SBAS
navigation message decoders and encoders.
"""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np


def parity_bit(x: int) -> int:
    """Return the odd-parity bit (1 if odd number of set bits, 0 otherwise).

    Parameters
    ----------
    x : int
        Integer value or bitfield.

    Returns
    -------
    int
        1 if popcount is odd, 0 if even.
    """
    v = int(x)
    if v < 0:
        v = v & 0xFFFFFFFF
    return v.bit_count() & 1


def unpack_uint(bits: Sequence[int] | np.ndarray, lo: int = 0, n: int | None = None) -> int:
    """Extract big-endian unsigned integer from a bit sequence.

    Parameters
    ----------
    bits : Sequence[int] | np.ndarray
        Bit array or list of {0, 1} bits.
    lo : int, default 0
        Starting bit offset.
    n : int | None, default None
        Number of bits to unpack. If None, unpacks to the end of bits.

    Returns
    -------
    int
        Unsigned integer value.
    """
    if n is None:
        n = len(bits) - lo
    if n <= 0 or lo >= len(bits):
        return 0
    v = 0
    for b in bits[lo : lo + n]:
        v = (v << 1) | (int(b) & 1)
    return v


def unpack_int(bits: Sequence[int] | np.ndarray, lo: int = 0, n: int | None = None) -> int:
    """Extract big-endian two's-complement signed integer from a bit sequence.

    Parameters
    ----------
    bits : Sequence[int] | np.ndarray
        Bit array or list of {0, 1} bits.
    lo : int, default 0
        Starting bit offset.
    n : int | None, default None
        Number of bits to unpack. If None, unpacks to the end of bits.

    Returns
    -------
    int
        Signed two's-complement integer value.
    """
    if n is None:
        n = len(bits) - lo
    if n <= 0 or lo >= len(bits):
        return 0
    v = unpack_uint(bits, lo, n)
    if v >= (1 << (n - 1)):
        v -= (1 << n)
    return v


def unpack_field(
    bits: Sequence[int] | np.ndarray,
    lo: int,
    kind: str,
    scale: float | None,
    n: int,
) -> float | int:
    """Extract and optionally scale a named field ('u' for unsigned, 's' for signed).

    Parameters
    ----------
    bits : Sequence[int] | np.ndarray
        Bit array or list.
    lo : int
        Starting bit offset in bits.
    kind : str
        's' for signed two's complement, 'u' for unsigned integer.
    scale : float | None
        Physical scale factor (LSB value). If None, unscaled integer is returned.
    n : int
        Field width in bits.

    Returns
    -------
    float | int
        Scaled float if scale is not None, or raw integer if scale is None.
    """
    v = unpack_int(bits, lo, n) if kind == "s" else unpack_uint(bits, lo, n)
    return v if scale is None else v * scale


def bits_of(value: int, n: int) -> list[int]:
    """Convert integer to big-endian list of n binary bits with two's-complement masking.

    Parameters
    ----------
    value : int
        Integer value (positive or negative).
    n : int
        Desired number of output bits.

    Returns
    -------
    list[int]
        List of n binary {0, 1} digits (MSB at index 0).
    """
    if n <= 0:
        return []
    v = int(value) & ((1 << n) - 1)
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def clean_float(x: object) -> float:
    """Convert input to float, sanitizing None, NaN, and Inf to 0.0.

    Parameters
    ----------
    x : object
        Input value to sanitize.

    Returns
    -------
    float
        Sanitized finite float value.
    """
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) or math.isinf(v) else v


def normalize_symbols(
    seq: Sequence[int | float] | np.ndarray,
    *,
    bpsk_sign: bool = False,
) -> list[int]:
    """Normalize input sequence into canonical {0, 1} integer bits.

    Handles bipolar {-1, 1} symbols as well as pre-sliced binary streams.

    Parameters
    ----------
    seq : Sequence[int | float] | np.ndarray
        Input symbols or bits.
    bpsk_sign : bool, default False
        If True, maps +1 -> 0 and -1 -> 1 (GPS LNAV BPSK convention).
        If False, maps +1 -> 1 and -1 -> 0 (Galileo / BeiDou / SBAS convention).

    Returns
    -------
    list[int]
        Canonical {0, 1} bit list.
    """
    arr = np.asarray(seq)
    if arr.size == 0:
        return []
    unique = np.unique(arr)
    if set(unique).issubset({-1, 1, -1.0, 1.0}):
        if bpsk_sign:
            return [(0 if x > 0 else 1) for x in arr]
        return [(1 if x > 0 else 0) for x in arr]
    return [int(x) & 1 for x in arr]


def deinterleave_block(
    sym: Sequence[int] | np.ndarray,
    rows: int,
    cols: int,
) -> list[int]:
    """Block deinterleaver: transpose rows x cols matrix written by rows and read by cols.

    Parameters
    ----------
    sym : Sequence[int] | np.ndarray
        Flat interleaved symbol sequence of length rows * cols.
    rows : int
        Number of rows in the interleaver block.
    cols : int
        Number of columns in the interleaver block.

    Returns
    -------
    list[int]
        Deinterleaved flat symbol list.
    """
    arr = np.asarray(sym)
    if arr.size != rows * cols:
        raise ValueError(
            f"Input size {arr.size} does not match dimensions {rows}x{cols} = {rows * cols}"
        )
    return arr.reshape(rows, cols).T.reshape(-1).tolist()
