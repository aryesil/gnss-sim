from __future__ import annotations

import datetime as dt
import json
import pathlib

import numpy as np
import pytest

from tests.synth.test_native_l1_multi_closed_loop_fix import l1_multi_run  # noqa: F401
import backend.inspector
from backend import config, geometry
from backend.analysis.l1_acquire import acquire_l1
from backend.scenario import ScenarioRequest
from backend.synth import engine

_MIXED = str(pathlib.Path(__file__).parent.parent / "fixtures" / "brdc_mixed.rnx")


def _wrap(err, n):
    return ((err + n / 2) % n) - n / 2


def test_acquire_qzss_sbas_galileo_beidou_match_engine(l1_multi_run):
    outdir, meta, expected = l1_multi_run
    band = next(b for b in meta["bands"] if b["id"] == "L1")
    iq = backend.inspector.read_iq(
        outdir / band["file"], "int16", max_samples=int(band["fs"] * 0.020)
    )
    tested_systems = set()
    for sv in meta["provenance"]["svs"]:
        if sv["sys"] not in ("J", "S", "E", "C", "G"):
            continue
        result = acquire_l1(
            iq, band["fs"], sv["sys"], sv["prn"], centre_offset_hz=sv["if_hz"]
        )
        exp_phase = expected[(sv["sys"], sv["prn"])] % sv["code_len"]
        phase_err = _wrap(result["code_phase_chips"] - exp_phase, sv["code_len"])
        assert result["metric_db"] > 15.0 and abs(phase_err) <= 0.1, (
            f"SV {sv['sys']}{sv['prn']}: metric={result['metric_db']:.2f} dB (want >15.0), "
            f"code_phase={result['code_phase_chips']:.4f}, expected={exp_phase:.4f}, "
            f"err={phase_err:.4f} chips (want <=0.1)"
        )
        tested_systems.add(sv["sys"])
    assert {"J", "S", "E", "C"}.issubset(tested_systems), (
        f"Missing required systems from tested set: {tested_systems}"
    )


@pytest.fixture(scope="module")
def gps_glo_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("gps_glo_fix")
    _orig_out = config.OUT_DIR
    config.OUT_DIR = out
    collected = []
    _orig_cm = geometry.constellation_multi

    def wrap(*a, **k):
        r = _orig_cm(*a, **k)
        collected.extend(r)
        return r

    geometry.constellation_multi = wrap
    try:
        req = ScenarioRequest(
            rinex_path=_MIXED,
            lat=41.0,
            lon=29.0,
            alt=100.0,
            start=dt.datetime(2026, 9, 1, 12, 0, 0),
            duration_s=1,
            sample_rate=20_000_000.0,
            sample_format="int16",
            engine="native",
            systems=("G", "R"),
            nav_message=False,
        )
        outdir = engine.run(req)
    finally:
        geometry.constellation_multi = _orig_cm
        config.OUT_DIR = _orig_out

    expected = {
        (e["sys"], e["prn"]): e["code_phase_chips"]
        for e in collected
        if "code_phase_chips" in e
    }
    meta_dict = json.loads((outdir / "meta.json").read_text())
    return outdir, meta_dict, expected


def test_acquire_gps_and_glonass_match_engine(gps_glo_run):
    outdir, meta, expected = gps_glo_run
    bands = {b["id"]: b for b in meta["bands"]}
    iq_by_sys = {
        "G": (
            backend.inspector.read_iq(
                outdir / bands["L1"]["file"],
                "int16",
                max_samples=int(bands["L1"]["fs"] * 0.020),
            ),
            bands["L1"]["fs"],
        ),
        "R": (
            backend.inspector.read_iq(
                outdir / bands["G1"]["file"],
                "int16",
                max_samples=int(bands["G1"]["fs"] * 0.020),
            ),
            bands["G1"]["fs"],
        ),
    }
    g_tested = 0
    r_tested = 0
    for sv in meta["provenance"]["svs"]:
        sysc = sv["sys"]
        if sysc not in ("G", "R"):
            continue
        iq, fs = iq_by_sys[sysc]
        if sysc == "R":
            result = acquire_l1(
                iq, fs, "R", sv["prn"], centre_offset_hz=0.0, glo_k=sv["glo_k"]
            )
            r_tested += 1
        else:
            result = acquire_l1(
                iq, fs, "G", sv["prn"], centre_offset_hz=sv["if_hz"]
            )
            g_tested += 1
        exp_phase = expected[(sv["sys"], sv["prn"])] % sv["code_len"]
        phase_err = _wrap(result["code_phase_chips"] - exp_phase, sv["code_len"])
        assert result["metric_db"] > 15.0 and abs(phase_err) <= 0.1, (
            f"SV {sv['sys']}{sv['prn']}: metric={result['metric_db']:.2f} dB (want >15.0), "
            f"code_phase={result['code_phase_chips']:.4f}, expected={exp_phase:.4f}, "
            f"err={phase_err:.4f} chips (want <=0.1)"
        )
    assert g_tested >= 1, f"Expected at least 1 G SV tested, got {g_tested}"
    assert r_tested >= 4, f"Expected at least 4 R SVs tested, got {r_tested}"


def test_acquire_unknown_system_raises():
    with pytest.raises(ValueError):
        acquire_l1(np.zeros(20000, np.complex64), 20e6, "X", 1, centre_offset_hz=0.0)


def test_acquire_glonass_requires_glo_k():
    with pytest.raises(ValueError):
        acquire_l1(np.zeros(20000, np.complex64), 20e6, "R", 1, centre_offset_hz=0.0)
