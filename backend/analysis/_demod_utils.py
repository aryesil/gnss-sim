"""Shared demodulation utilities for GNSS correlator signal processing.

Provides carrier phase de-rotation via Costas squaring and polynomial fitting.
"""
from __future__ import annotations

import numpy as np


def derotate_bpsk_carrier(seg: np.ndarray, deg: int = 2) -> np.ndarray:
    """De-rotate residual carrier phase from complex correlator prompt symbols.

    Employs Costas squaring (seg ** 2) to eliminate BPSK +/-1 data modulation,
    unwraps the resulting doubled phase trajectory, fits a degree-N polynomial
    to capture residual Doppler frequency and Doppler rate, and removes the
    estimated phase trajectory and residual constant phase.

    Parameters
    ----------
    seg : np.ndarray
        1D complex array of prompt correlator accumulations.
    deg : int, default 2
        Maximum polynomial fit degree (1 for residual frequency, 2 for frequency
        and Doppler acceleration rate). Clamped safely for small sample sizes.

    Returns
    -------
    np.ndarray
        Phase-corrected complex symbol array.
    """
    nsym = seg.size
    if nsym < 4:
        return seg

    d = min(deg, nsym - 1)
    if nsym < 8 and deg <= 2:
        d = 1 if nsym < 5 else min(d, 2)

    bi = np.arange(nsym, dtype=np.float64)
    ph = np.unwrap(np.angle(seg ** 2))
    coef = np.polyfit(bi, ph, d) / 2.0
    seg_derot = seg * np.exp(-1j * np.polyval(coef, bi))
    phi0 = 0.5 * np.angle(np.sum(seg_derot ** 2))
    return seg_derot * np.exp(-1j * phi0)
