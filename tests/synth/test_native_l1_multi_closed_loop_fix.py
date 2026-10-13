"""End-to-end multi-GNSS L1 closed-loop test: native engine synthesizes
QZSS, SBAS, Galileo E1, and BeiDou B1I alongside GPS on the L1 composite band;
the receiver solves a position fix from the composite IQ.
"""
import datetime as dt
import json
import pathlib

import pytest

from backend import config, geometry
from backend.analysis import receiver
from backend.ephem import ephemeris
from backend.scenario import ScenarioRequest
from backend.synth import engine

_MIXED = str(pathlib.Path(__file__).parent.parent / "fixtures" / "brdc_mixed.rnx")
_RX = (45.0, 30.0, 100.0)
_FS = 20_000_000.0
_DUR = 1
_START = dt.datetime(2026, 9, 1, 12, 0, 0)

_CAPTURED = {}


@pytest.fixture(scope="module")
def l1_multi_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("l1_multi_fix")
    _orig_out = config.OUT_DIR
    config.OUT_DIR = out
    collected = []
    _orig_cm = geometry.constellation_multi

    def wrap(*a, **k):
        if "eph" not in _CAPTURED and a:
            _CAPTURED["eph"] = a[0]
        r = _orig_cm(*a, **k)
        collected.extend(r)
        return r

    geometry.constellation_multi = wrap
    try:
        req = ScenarioRequest(
            rinex_path=_MIXED,
            lat=_RX[0],
            lon=_RX[1],
            alt=_RX[2],
            start=_START,
            duration_s=_DUR,
            sample_rate=_FS,
            sample_format="int16",
            engine="native",
            systems=("G", "J", "E", "C", "S"),
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
    approx_sow = ephemeris.gps_week_and_sow(
        _START + dt.timedelta(seconds=config.GPS_UTC_LEAP_S)
    )[1]
    _CAPTURED["entries"] = collected
    _CAPTURED["sow"] = approx_sow
    return outdir, meta_dict, expected


def test_fixture_has_enough_svs(l1_multi_run):
    _outdir, meta, _expected = l1_multi_run
    svs = [s for s in meta["provenance"]["svs"] if s["sys"] != "R"]
    counts = {}
    for s in svs:
        counts[s["sys"]] = counts.get(s["sys"], 0) + 1
    assert len(svs) >= 5, (
        f"Expected >= 5 non-GLONASS SVs in fixture, got {len(svs)}. "
        f"Per-system counts: {counts}"
    )


@pytest.mark.xfail(strict=True, reason="multi-GNSS L1 receiver missing")
def test_l1_multi_fix_within_10m(l1_multi_run):
    outdir, _meta, _expected = l1_multi_run
    gps_start = _START + dt.timedelta(seconds=config.GPS_UTC_LEAP_S)
    _week, approx_time_gps = ephemeris.gps_week_and_sow(gps_start)
    eph = {}
    result = receiver.fix_from_iq(
        outdir / "gpssim.bin",
        "int16",
        20_000_000.0,
        eph,
        approx_time_gps,
        marker_llh=_RX,
        band="L1",
    )
    assert "error" not in result, f"receiver returned error: {result}"
    assert result["error_m"] < 10.0, f"error_m={result.get('error_m')}"
