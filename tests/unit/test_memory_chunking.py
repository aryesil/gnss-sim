"""Unit tests for demodulator memory chunking (< 50 MB) and package cleanliness."""
from __future__ import annotations

import pathlib
import tracemalloc
import numpy as np
import pytest

from backend.analysis import bds_d1_decode, inav_decode, lnav_decode, sbas_decode


def test_lnav_prompt_per_ms_memory_bound():
    """Verify LNAV prompt correlation on 52M samples uses < 50 MB peak RAM."""
    fs = 2_600_000.0
    N = int(fs * 20.0)  # 52M samples
    x_wiped = np.zeros(N, dtype=np.complex64)
    ca = np.ones(1023, dtype=np.float64)

    tracemalloc.start()
    tracemalloc.reset_peak()
    out = lnav_decode._prompt_per_ms(x_wiped, fs, ca, 0.0, 1.023e6)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    assert len(out) == 20000
    assert peak_mb < 50.0, f"LNAV peak memory {peak_mb:.2f} MB exceeded 50 MB bound"


def test_lnav_prompt_per_ms_numerical_exactness():
    """Verify chunked prompt correlation matches monolithic calculation on non-trivial signal."""
    fs = 2_600_000.0
    N = int(fs * 0.2)  # 200 ms
    np.random.seed(42)
    x_wiped = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)
    ca = np.random.choice([-1.0, 1.0], size=1023)

    out_chunked = lnav_decode._prompt_per_ms(x_wiped, fs, ca, 12.5, 1.023005e6, chunk_ms=20)
    out_monolithic = lnav_decode._prompt_per_ms(x_wiped, fs, ca, 12.5, 1.023005e6, chunk_ms=1000)

    np.testing.assert_allclose(out_chunked, out_monolithic, rtol=1e-5, atol=1e-8)


def test_inav_demod_symbols_memory_bound():
    """Verify Galileo E1B demodulation on 8M samples uses < 50 MB peak RAM."""
    fs = 4_092_000.0
    N = int(fs * 2.0)  # 8.18M samples
    iq = np.zeros(N, dtype=np.complex64)

    tracemalloc.start()
    tracemalloc.reset_peak()
    syms = inav_decode.demod_symbols_e1b(iq, fs, prn=5, dopp_hz=1500.0, code_phase_chips=0.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    assert len(syms) == 500
    assert peak_mb < 50.0, f"INAV peak memory {peak_mb:.2f} MB exceeded 50 MB bound"


def test_bds_d1_demod_symbols_memory_bound():
    """Verify BeiDou B1I demodulation on 8M samples uses < 50 MB peak RAM."""
    fs = 4_092_000.0
    N = int(fs * 2.0)
    iq = np.zeros(N, dtype=np.complex64)

    tracemalloc.start()
    tracemalloc.reset_peak()
    syms = bds_d1_decode.demod_symbols_b1i(iq, fs, prn=6, dopp_hz=1500.0, code_phase_chips=0.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 50.0, f"BDS D1 peak memory {peak_mb:.2f} MB exceeded 50 MB bound"


def test_sbas_demod_symbols_memory_bound():
    """Verify SBAS L1 demodulation on 8M samples uses < 50 MB peak RAM."""
    fs = 4_092_000.0
    N = int(fs * 2.0)
    iq = np.zeros(N, dtype=np.complex64)

    tracemalloc.start()
    tracemalloc.reset_peak()
    syms = sbas_decode.demod_symbols_sbas_l1(iq, fs, prn=120, dopp_hz=1200.0, code_phase_chips=0.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 50.0, f"SBAS peak memory {peak_mb:.2f} MB exceeded 50 MB bound"


def test_backend_analysis_init_clean():
    """Verify backend/analysis/__init__.py contains no hardcoded absolute staging paths."""
    import backend.analysis as ba
    init_file = pathlib.Path(ba.__file__)
    content = init_file.read_text(encoding="utf-8")
    assert "D:\\" not in content and "D:/" not in content
    assert "target_path" not in content
