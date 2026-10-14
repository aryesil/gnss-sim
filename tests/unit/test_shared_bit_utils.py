"""Unit tests for backend.analysis._bit_utils, _crc, and _demod_utils shared primitives."""
from __future__ import annotations

import math
import numpy as np
import pytest

from backend.analysis._bit_utils import (
    bits_of,
    clean_float,
    deinterleave_block,
    normalize_symbols,
    parity_bit,
    unpack_field,
    unpack_int,
    unpack_uint,
)
from backend.analysis._crc import check_crc24q, crc24q
from backend.analysis._demod_utils import derotate_bpsk_carrier


def test_parity_bit_primitives():
    # Powers of two have 1 bit set -> parity 1
    for shift in range(32):
        assert parity_bit(1 << shift) == 1
    # Zero has 0 bits set -> parity 0
    assert parity_bit(0) == 0
    # Even popcount -> parity 0
    assert parity_bit(0b11) == 0
    assert parity_bit(0b1111) == 0
    # Standard GNSS convolutional polynomials
    assert parity_bit(0o171) == (bin(0o171).count("1") & 1)
    assert parity_bit(0o133) == (bin(0o133).count("1") & 1)
    # Negative values
    assert parity_bit(-1) == 0  # 32 bits of 1s in 32-bit mask -> 32 & 1 == 0


def test_unpack_uint_variations():
    # Single list argument
    assert unpack_uint([1, 0, 1]) == 5
    assert unpack_uint([0, 0, 0, 0]) == 0
    assert unpack_uint([1] * 16) == 65535
    # Offsets and widths
    assert unpack_uint([0, 1, 0, 1, 1], lo=1, n=3) == 0b101  # 5
    assert unpack_uint([1, 1, 0, 0], lo=2, n=2) == 0
    # NumPy arrays
    arr = np.array([1, 0, 0, 1], dtype=np.int8)
    assert unpack_uint(arr) == 9
    assert unpack_uint(arr, lo=1, n=2) == 0
    # Empty and boundary cases
    assert unpack_uint([]) == 0
    assert unpack_uint([1, 1], lo=2, n=2) == 0
    assert unpack_uint([1, 1], lo=0, n=0) == 0
    assert unpack_uint([1, 1], lo=5, n=2) == 0


def test_unpack_int_twos_complement():
    # Positive values
    assert unpack_int([0, 1, 0, 1]) == 5
    assert unpack_int([0, 0, 0, 1], lo=0, n=4) == 1
    # Negative values
    assert unpack_int([1]) == -1
    assert unpack_int([1, 1, 1, 1]) == -1
    assert unpack_int([1, 0, 0, 0]) == -8
    assert unpack_int([0, 1, 1, 1]) == 7
    # Boundary: min/max 8-bit
    assert unpack_int([1, 0, 0, 0, 0, 0, 0, 0]) == -128
    assert unpack_int([0, 1, 1, 1, 1, 1, 1, 1]) == 127
    # Offsets
    stream = [0, 1, 1, 1, 0, 0]
    assert unpack_int(stream, lo=1, n=3) == -1  # 111_2 in 3 bits is -1
    # Empty
    assert unpack_int([], 0, 0) == 0


def test_unpack_field_scaling():
    bits = [0, 1, 0, 0, 1, 1, 1, 1]
    # Unsigned scaled
    assert unpack_field(bits, 0, "u", 0.5, 4) == pytest.approx(2.0)
    # Signed scaled
    assert unpack_field(bits, 4, "s", 0.25, 4) == pytest.approx(-0.25)
    # Unscaled (scale=None) -> returns integer
    res_int = unpack_field(bits, 0, "u", None, 4)
    assert res_int == 4 and isinstance(res_int, int)
    # Scale 1.0 -> returns float
    res_float = unpack_field(bits, 0, "u", 1.0, 4)
    assert res_float == pytest.approx(4.0) and isinstance(res_float, float)


def test_bits_of_ordering_and_masking():
    assert bits_of(5, 4) == [0, 1, 0, 1]
    assert bits_of(0, 4) == [0, 0, 0, 0]
    # Negative number masked to two's complement
    assert bits_of(-1, 4) == [1, 1, 1, 1]
    assert bits_of(-8, 4) == [1, 0, 0, 0]
    # Overflow truncation
    assert bits_of(0x1F, 4) == [1, 1, 1, 1]
    assert len(bits_of(42, 12)) == 12
    # Zero or negative width
    assert bits_of(42, 0) == []
    assert bits_of(42, -5) == []


def test_clean_float_sanitization():
    assert clean_float(3.14159) == pytest.approx(3.14159)
    assert clean_float(42) == 42.0
    assert clean_float("1.25") == 1.25
    assert clean_float(None) == 0.0
    assert clean_float(float("nan")) == 0.0
    assert clean_float(np.nan) == 0.0
    assert clean_float(float("inf")) == 0.0
    assert clean_float(float("-inf")) == 0.0
    assert clean_float("invalid") == 0.0


def test_normalize_symbols():
    # Canonical bits
    assert normalize_symbols([0, 1, 1, 0]) == [0, 1, 1, 0]
    # Bipolar {-1, 1} symbols with bpsk_sign=False (+1 -> 1, -1 -> 0)
    assert normalize_symbols([1, -1, 1, -1], bpsk_sign=False) == [1, 0, 1, 0]
    # Bipolar {-1, 1} symbols with bpsk_sign=True (+1 -> 0, -1 -> 1)
    assert normalize_symbols([1, -1, 1, -1], bpsk_sign=True) == [0, 1, 0, 1]
    # NumPy array
    arr = np.array([-1.0, 1.0, 1.0])
    assert normalize_symbols(arr, bpsk_sign=False) == [0, 1, 1]
    # Empty
    assert normalize_symbols([]) == []


def test_deinterleave_block():
    raw = list(range(12))
    expected = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    assert deinterleave_block(raw, rows=4, cols=3) == expected

    # Dimension mismatch raises ValueError
    with pytest.raises(ValueError, match="Input size 11 does not match"):
        deinterleave_block(list(range(11)), rows=4, cols=3)


def test_crc24q_standard_vectors():
    # Standard ASCII test vector '123456789'
    ascii_bits = [(c >> (7 - i)) & 1 for c in b"123456789" for i in range(8)]
    v = crc24q(ascii_bits)
    assert hex(int("".join(map(str, v)), 2)) == "0xc0f7af"

    # Zero sequence
    assert crc24q([0] * 200) == [0] * 24

    # Single bit 1
    assert hex(int("".join(map(str, crc24q([1]))), 2)) == "0x360952"

    # Non-byte aligned lengths
    gal_bits = [1, 0] * 98  # 196 bits
    crc_gal = crc24q(gal_bits)
    assert len(crc_gal) == 24
    assert check_crc24q(gal_bits + crc_gal, data_bits=196, crc_bits=24) is True
    # Corrupt 1 bit
    corrupt = list(gal_bits + crc_gal)
    corrupt[50] ^= 1
    assert check_crc24q(corrupt, data_bits=196, crc_bits=24) is False


def test_check_crc24q_bounds():
    # Too short
    assert check_crc24q([1, 0, 1], data_bits=10, crc_bits=24) is False


def test_derotate_bpsk_carrier():
    # Less than 4 samples returns unmodified
    short = np.array([1.0 + 0j, -1.0 + 0j])
    assert np.all(derotate_bpsk_carrier(short) == short)

    # Clean BPSK with constant phase rotation
    nsym = 20
    bpsk = np.array([1.0 if (i % 3 == 0) else -1.0 for i in range(nsym)], dtype=np.complex128)
    theta = np.pi / 4.0  # 45 deg rotation
    rotated = bpsk * np.exp(1j * theta)
    derot = derotate_bpsk_carrier(rotated, deg=2)
    # Energy should concentrate heavily on real axis
    real_power = np.sum(np.real(derot) ** 2)
    imag_power = np.sum(np.imag(derot) ** 2)
    assert real_power > 10.0 * imag_power
