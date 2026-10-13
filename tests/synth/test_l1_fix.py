from __future__ import annotations

import copy
import pytest

import backend.inspector
from backend.analysis.l1_fix import default_atmo_fn, fix_from_iq_l1_multi
from tests.synth.test_native_l1_multi_closed_loop_fix import (  # noqa: F401
    l1_multi_run,
    _CAPTURED,
    _MIXED,
    _RX,
)


def _run(l1_multi_run, meta_override=None, atmo=True):
    outdir, meta, expected = l1_multi_run
    meta = meta_override or meta
    iq_by_band = {}
    for b in meta["bands"]:
        iq_by_band[b["id"]] = (
            backend.inspector.read_iq(
                outdir / b["file"], "int16", max_samples=int(b["fs"] * 0.020)
            ),
            b["fs"],
        )
    eph = _CAPTURED["eph"]
    sow = _CAPTURED["sow"]
    atmo_fn = (
        default_atmo_fn(rinex_path=_MIXED, approx_llh=_RX, t_gps_sow=sow)
        if atmo
        else None
    )
    return fix_from_iq_l1_multi(
        iq_by_band,
        meta,
        eph,
        sow,
        approx_llh=_RX,
        marker_llh=_RX,
        atmo_fn=atmo_fn,
    )


def test_measured_pseudorange_matches_engine(l1_multi_run):
    result = _run(l1_multi_run)
    assert "error" not in result, f"Fix failed: {result}"
    compared = 0
    for e in _CAPTURED["entries"]:
        key = f"{e['sys']}{e['prn']}"
        if key in result.get("per_sv", {}):
            meas = result["per_sv"][key]["measured_pr_m"]
            exp = e["pseudorange_m"]
            assert abs(meas - exp) <= 10.0, (
                f"SV {key}: measured_pr_m={meas}, engine pseudorange_m={exp}"
            )
            compared += 1
    assert compared >= 5, f"Only compared {compared} SVs (want >= 5)"


def test_corrected_pseudorange_equals_geometric_range(l1_multi_run):
    result = _run(l1_multi_run)
    assert "error" not in result, f"Fix failed: {result}"
    compared = 0
    for e in _CAPTURED["entries"]:
        key = f"{e['sys']}{e['prn']}"
        if key in result.get("per_sv", {}):
            corr = result["per_sv"][key]["corrected_pr_m"]
            geo = e["geo_range_m"]
            assert abs(corr - geo) <= 15.0, (
                f"SV {key}: corrected_pr_m={corr}, engine geo_range_m={geo}"
            )
            compared += 1
    assert compared >= 5, f"Only compared {compared} SVs (want >= 5)"


def test_fix_within_30m(l1_multi_run):
    result = _run(l1_multi_run)
    assert "error" not in result, f"Fix returned error: {result}"
    assert result["error_m"] < 30.0, f"error_m={result.get('error_m')}"
    assert isinstance(result["clock_bias_s"], float)
    assert len(result["prns_used"]) >= 5
    print(
        f"\n[test_fix_within_30m] error_m={result['error_m']:.4f}, "
        f"residual_rms_m={result['residual_rms_m']:.4f}, "
        f"pdop={result['pdop']:.4f}"
    )


def test_too_few_svs_returns_error(l1_multi_run):
    outdir, meta, expected = l1_multi_run
    meta_mod = copy.deepcopy(meta)
    meta_mod["provenance"]["svs"] = meta_mod["provenance"]["svs"][:3]
    result = _run(l1_multi_run, meta_override=meta_mod)
    assert "error" in result, f"Expected error in result, got: {result}"
    assert "prns_used" in result


def test_glonass_without_glo_k_is_skipped_and_named(l1_multi_run):
    outdir, meta, expected = l1_multi_run
    meta_mod = copy.deepcopy(meta)
    meta_mod["provenance"]["svs"].append(
        {
            "sys": "R",
            "prn": 1,
            "band": "L1",
            "code_len": 511,
            "chip_hz": 511000.0,
            "if_hz": 0.0,
        }
    )
    result = _run(l1_multi_run, meta_override=meta_mod)
    assert "skipped" in result
    assert "R1" in result["skipped"]
    assert "glo_k" in result["skipped"]["R1"]
    assert "error" not in result, f"Expected fix to succeed, got: {result}"
