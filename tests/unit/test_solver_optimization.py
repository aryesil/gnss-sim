"""Unit tests for PVT solver conditioning, speed, and coordinate deduplication."""
from __future__ import annotations

import numpy as np
import pytest

from backend import config, geometry
from backend.analysis import receiver


def _sample_constellation():
    """Return a standard 6-satellite constellation and ground truth receiver point."""
    rx_true = np.array([3794425.0, 2138125.0, 4712433.0], dtype=np.float64)  # approx Istanbul
    clock_bias_true_s = 1.25e-6  # 1.25 microseconds = 374.74 m
    sat_pos = {
        "G01": np.array([15600000.0, 12000000.0, 18000000.0], dtype=np.float64),
        "G02": np.array([-12000000.0, 18000000.0, 16000000.0], dtype=np.float64),
        "G03": np.array([8000000.0, -16000000.0, 20000000.0], dtype=np.float64),
        "G04": np.array([20000000.0, -8000000.0, 14000000.0], dtype=np.float64),
        "G05": np.array([-18000000.0, -10000000.0, 17000000.0], dtype=np.float64),
        "G06": np.array([5000000.0, 21000000.0, 15000000.0], dtype=np.float64),
    }
    pseudoranges = {}
    for prn, spos in sat_pos.items():
        geometric_range = float(np.linalg.norm(spos - rx_true))
        pseudoranges[prn] = geometric_range + config.C * clock_bias_true_s
    return rx_true, clock_bias_true_s, sat_pos, pseudoranges


def test_condition_number_reduction():
    """Verify condition number reduction from column scaling (c * dt -> dt_m)."""
    rx_true, _, sat_pos, _ = _sample_constellation()
    prns = sorted(sat_pos)
    S = np.array([sat_pos[p] for p in prns], dtype=np.float64)
    rng = np.linalg.norm(S - rx_true, axis=1)

    # Legacy unscaled design matrix H
    H_unscaled = np.column_stack([(rx_true - S) / rng[:, None], config.C * np.ones(len(prns))])
    cond_unscaled = np.linalg.cond(H_unscaled)

    # Scaled design matrix H
    H_scaled = np.column_stack([(rx_true - S) / rng[:, None], np.ones(len(prns))])
    cond_scaled = np.linalg.cond(H_scaled)

    # Scaled condition number must be < 50 while unscaled is > 1e9
    assert cond_unscaled > 1e9, f"Expected unscaled cond > 1e9, got {cond_unscaled}"
    assert cond_scaled < 50.0, f"Expected scaled cond < 50, got {cond_scaled}"
    assert cond_unscaled / cond_scaled > 1e7


def test_solve_position_known_point_convergence():
    """Verify solve_position accurately recovers ECEF point and clock bias."""
    rx_true, clock_bias_true_s, sat_pos, pseudoranges = _sample_constellation()
    sol = receiver.solve_position(pseudoranges, sat_pos)

    assert "ecef" in sol
    assert "clock_bias_s" in sol
    assert "iterations" in sol
    assert "residual_rms_m" in sol

    pos_err = np.linalg.norm(np.array(sol["ecef"]) - rx_true)
    assert pos_err < 1e-4, f"Position error {pos_err} m exceeded 0.1 mm"

    clk_err = abs(sol["clock_bias_s"] - clock_bias_true_s)
    assert clk_err < 1e-12, f"Clock bias error {clk_err} s exceeded 1 ps"

    assert sol["iterations"] <= 6
    assert sol["residual_rms_m"] < 1e-4


def test_solve_position_with_initial_guess():
    """Verify solve_position accepts and converges properly from x0."""
    rx_true, clock_bias_true_s, sat_pos, pseudoranges = _sample_constellation()
    x0 = [rx_true[0] + 500.0, rx_true[1] - 500.0, rx_true[2] + 200.0, clock_bias_true_s + 1e-7]
    sol = receiver.solve_position(pseudoranges, sat_pos, x0=x0)

    pos_err = np.linalg.norm(np.array(sol["ecef"]) - rx_true)
    assert pos_err < 1e-4
    assert abs(sol["clock_bias_s"] - clock_bias_true_s) < 1e-12


def test_solve_position_few_satellites_raises():
    """Verify solve_position raises ValueError with < 4 satellites."""
    sat_pos = {
        "G01": [1e7, 1e7, 2e7],
        "G02": [-1e7, 1e7, 2e7],
        "G03": [1e7, -1e7, 2e7],
    }
    pr = {"G01": 2e7, "G02": 2e7, "G03": 2e7}
    with pytest.raises(ValueError, match="need >= 4 satellites"):
        receiver.solve_position(pr, sat_pos)


def test_solve_position_singular_fallback():
    """Verify fallback to lstsq on rank-deficient / singular geometry."""
    # 4 coplanar satellites where normal equations matrix N is singular
    sat_pos = {
        "G01": [1e7, 0.0, 2e7],
        "G02": [2e7, 0.0, 2e7],
        "G03": [3e7, 0.0, 2e7],
        "G04": [4e7, 0.0, 2e7],
    }
    pr = {"G01": 2e7, "G02": 2.1e7, "G03": 2.2e7, "G04": 2.3e7}
    # Should not raise LinAlgError due to except np.linalg.LinAlgError: dX, *_ = np.linalg.lstsq(...)
    sol = receiver.solve_position(pr, sat_pos)
    assert isinstance(sol["ecef"], list)
    assert len(sol["ecef"]) == 3


def test_ecef_to_llh_delegation():
    """Verify _ecef_to_llh in receiver delegates to geometry.ecef_to_llh identically."""
    test_coords = [
        (3794425.0, 2138125.0, 4712433.0),
        (6378137.0, 0.0, 0.0),
        (0.0, 6378137.0, 0.0),
        (0.0, 0.0, 6356752.314),
        (-2694044.0, -4293642.0, 3854776.0),
    ]
    for x, y, z in test_coords:
        lat1, lon1, h1 = receiver._ecef_to_llh(x, y, z)
        lat2, lon2, h2 = geometry.ecef_to_llh(x, y, z)
        assert lat1 == pytest.approx(lat2, abs=1e-11)
        assert lon1 == pytest.approx(lon2, abs=1e-11)
        assert h1 == pytest.approx(h2, abs=1e-9)
