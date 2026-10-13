# tests/test_receiver.py
import numpy as np

from backend.analysis import receiver
from backend import config, geometry
from tests.synth.test_native_l1_multi_closed_loop_fix import (  # noqa: F401
    l1_multi_run,
)


def test_solve_position_recovers_known_point():
    rng = np.random.default_rng(1)
    true = np.array(geometry.llh_to_ecef(41.0, 29.0, 120.0))
    b_true = 3.5e-4
    up = true / np.linalg.norm(true)
    dirs = [[0.05, 0.02, 0.3], [0.6, -0.1, 0.2], [-0.5, 0.4, 0.15],
            [0.1, -0.6, 0.25], [-0.3, -0.35, 0.1], [0.45, 0.5, 0.05]]
    sats, pr = {}, {}
    for i, d in enumerate(dirs, start=1):
        v = up + np.array(d)
        s = v / np.linalg.norm(v) * 26.56e6
        sats[i] = s
        pr[i] = np.linalg.norm(s - true) + 299792458.0 * b_true + rng.normal(0, 3)
    out = receiver.solve_position(pr, sats)
    assert np.linalg.norm(np.array(out["ecef"]) - true) < 30.0
    assert abs(out["clock_bias_s"] - b_true) < 1e-7
    assert out["iterations"] < 10


def test_fix_from_iq_band_param_guard():
    import inspect
    import pytest

    sig = inspect.signature(receiver.fix_from_iq)
    p = sig.parameters["band"]
    assert p.default == "L1"
    assert p.kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(ValueError):
        receiver.fix_from_iq("/nonexistent.bin", "int8", 5e6, {}, 0.0,
                             band="L9")


def test_l5_blind_scan_acquires_nothing_on_pure_noise(tmp_path):
    # The largest of the ~3M noise-only acquisition cells sits ~13-15 dB
    # above the median; a threshold below that "acquires" every scanned
    # PRN and sends each one into a full nav demod.
    rng = np.random.default_rng(7)
    raw = np.clip(np.round(rng.standard_normal(2 * 75_000) * 20), -127, 127)
    path = tmp_path / "noise_l5.bin"
    raw.astype(np.int8).tofile(path)
    res = receiver.fix_from_iq(path, "int8", 25e6, {}, 0.0, band="L5")
    assert res["nav_decode"] == {}


def test_gps_only_unchanged(tmp_path):
    import inspect

    sig = inspect.signature(receiver.fix_from_iq)
    assert "systems" in sig.parameters
    assert sig.parameters["systems"].default is None

    rng = np.random.default_rng(42)
    raw = np.clip(np.round(rng.standard_normal(2 * 50_000) * 20), -127, 127)
    path = tmp_path / "noise_l1.bin"
    raw.astype(np.int8).tofile(path)

    res_none = receiver.fix_from_iq(path, "int8", 2.5e6, {}, 0.0, band="L1")
    res_g = receiver.fix_from_iq(path, "int8", 2.5e6, {}, 0.0, band="L1", systems=["G"])
    assert res_none == res_g
    assert res_g == {"error": "only 0 PRNs acquired", "prns_used": []}


def test_run_receiver_multi_endpoint(monkeypatch, l1_multi_run):
    from fastapi.testclient import TestClient
    from backend.app import app

    outdir, meta, _expected = l1_multi_run
    monkeypatch.setattr(config, "OUT_DIR", outdir.parent)
    client = TestClient(app)
    resp = client.post(
        "/api/receiver",
        json={"outdir": outdir.name, "marker": [45.0, 30.0, 100.0]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "error" not in data, f"receiver returned error: {data}"
    assert data["error_m"] < 30.0, f"error_m={data.get('error_m')}"

