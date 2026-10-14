"""Unit tests for shared vectorized rate-1/2 K=7 Viterbi decoding engine."""
from __future__ import annotations

import time
import numpy as np
import pytest

from backend.analysis._viterbi import (
    viterbi_decode,
    viterbi_free_decode,
    _G1,
    _G2,
    _NXT_INV,
    _OUT_INV,
    _NXT_STD,
    _OUT_STD,
)
import backend.analysis.inav_encode as inav_enc
import backend.analysis.sbas_encode as sbas_enc
import backend.analysis.cnav_encode as cnav_enc


class TestViterbiFreeDecode:
    def test_inav_clean_roundtrip(self):
        data = [1, 0, 1, 1, 0, 0, 1, 0] * 15  # 120 bits
        coded = inav_enc.conv_encode(data)
        decoded = viterbi_free_decode(coded, invert_g2=True)
        assert decoded == data

    def test_sbas_clean_roundtrip(self):
        conv = sbas_enc._Conv()
        data = [1, 1, 0, 0, 1, 0, 1, 0] * 25  # 200 bits
        coded = conv.encode(data)
        decoded = viterbi_free_decode(coded, invert_g2=False)
        assert decoded == data

    def test_cnav_clean_roundtrip(self):
        conv = cnav_enc._Conv()
        data = [0, 1, 1, 0, 1, 0, 0, 1] * 20  # 160 bits
        coded = conv.encode(data)
        decoded = viterbi_free_decode(coded, invert_g2=True)
        assert decoded == data

    def test_empty_and_odd_inputs(self):
        assert viterbi_free_decode([]) == []
        assert viterbi_free_decode([1]) == []
        assert len(viterbi_free_decode([1, 0, 1])) == 1


class TestViterbiDecodeTrellis:
    def test_inav_constrained_start_clean(self):
        data = [1, 0, 0, 1, 1, 1, 0, 0] * 15
        coded = inav_enc.conv_encode(data)
        decoded = viterbi_decode(coded, invert_g2=True, known_start_state=0)
        assert decoded == data

    def test_inav_corrects_single_error(self):
        data = [1, 0, 1, 0, 0, 1, 1, 0] * 15
        coded = inav_enc.conv_encode(data)
        coded_err = list(coded)
        coded_err[45] ^= 1  # Invert one bit
        decoded = viterbi_decode(coded_err, invert_g2=True, known_start_state=0)
        assert decoded == data

    def test_sbas_unconstrained_start_clean(self):
        conv = sbas_enc._Conv()
        data = [1, 0, 1, 1, 0, 0, 1, 0] * 25
        coded = conv.encode(data)
        decoded = viterbi_decode(coded, invert_g2=False, known_start_state=None)
        assert decoded == data

    def test_sbas_corrects_single_error(self):
        conv = sbas_enc._Conv()
        data = [1, 1, 0, 1, 0, 0, 1, 0] * 25
        coded = conv.encode(data)
        coded_err = list(coded)
        coded_err[80] ^= 1
        decoded = viterbi_decode(coded_err, invert_g2=False, known_start_state=None)
        assert decoded == data

    def test_cnav_unconstrained_start_clean(self):
        conv = cnav_enc._Conv()
        data = [0, 1, 1, 0, 1, 1, 0, 0] * 20
        coded = conv.encode(data)
        decoded = viterbi_decode(coded, invert_g2=True, known_start_state=None)
        assert decoded == data

    def test_empty_and_short_inputs(self):
        assert viterbi_decode([]) == []
        assert viterbi_decode([1]) == []
        assert len(viterbi_decode([0, 1, 1])) == 1

    def test_numpy_and_bool_inputs(self):
        data = [1, 0, 1, 1, 0, 0, 1, 0] * 10
        coded = inav_enc.conv_encode(data)
        arr_float = np.array(coded, dtype=np.float64)
        arr_bool = np.array(coded, dtype=bool)
        assert viterbi_decode(arr_float, invert_g2=True, known_start_state=0) == data
        assert viterbi_decode(arr_bool, invert_g2=True, known_start_state=0) == data


class TestViterbiPerformance:
    def test_vectorized_latency_benchmark(self):
        """Verify that decoding 120-bit INAV frame completes in < 1.0 ms."""
        data = [1, 0, 1, 1, 0, 0, 1, 0] * 15
        coded = inav_enc.conv_encode(data)
        # Warmup
        for _ in range(5):
            viterbi_decode(coded, invert_g2=True, known_start_state=0)

        t0 = time.perf_counter()
        n_iters = 50
        for _ in range(n_iters):
            viterbi_decode(coded, invert_g2=True, known_start_state=0)
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) / n_iters * 1000.0
        # Typical execution is ~0.37 ms; assert well below 1.0 ms (legacy was 1.9 ms)
        assert avg_ms < 1.0, f"Viterbi decode took {avg_ms:.3f} ms (expected < 1.0 ms)"
