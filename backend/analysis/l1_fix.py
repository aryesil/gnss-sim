"""Multi-GNSS L1 code-phase receiver position fix.

Computes pseudoranges from L1-group acquisitions across GPS, QZSS, SBAS,
Galileo, BeiDou, and GLONASS, applies satellite clock, group delay bias,
and atmospheric corrections, and solves for receiver position.
"""
from __future__ import annotations

import math
import numpy as np

from backend import config, geometry
from backend.analysis.l1_acquire import acquire_l1
from backend.analysis.receiver import _ecef_to_llh, solve_position
from backend.ephem import ephemeris
from backend.models import atmosphere
from backend.synth import engine, signals


def default_atmo_fn(rinex_path, approx_llh, t_gps_sow):
    lat_deg, lon_deg, h_m = approx_llh
    cfg = atmosphere.AtmosphereConfig.from_dict(engine._DEFAULT_ATMOSPHERE)
    try:
        hdr = dict(ephemeris.rinex_header_iono_utc(rinex_path) or {})
    except Exception:
        hdr = {}
    if cfg.ionosphere == "klobuchar" and hdr.get("iono_alpha") and hdr.get("iono_beta"):
        cfg.klobuchar_alpha = tuple(float(x) for x in hdr["iono_alpha"][:4])
        cfg.klobuchar_beta = tuple(float(x) for x in hdr["iono_beta"][:4])

    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    def atmo_delay(az_rad: float, el_rad: float, carrier_hz: float) -> float:
        iono = (
            atmosphere.klobuchar_delay_m(
                cfg.klobuchar_alpha,
                cfg.klobuchar_beta,
                t_gps_sow,
                lat_rad,
                lon_rad,
                az_rad,
                el_rad,
                freq_hz=carrier_hz,
            )["delay_m"]
            if cfg.ionosphere == "klobuchar"
            else 0.0
        )
        tropo = (
            atmosphere.saastamoinen_delay_m(
                el_rad,
                h_m,
                cfg.pressure_hpa,
                cfg.temperature_k,
                cfg.humidity,
            )["delay_m"]
            if cfg.troposphere == "saastamoinen"
            else 0.0
        )
        return float(iono + tropo)

    return atmo_delay


def fix_from_iq_l1_multi(
    iq_by_band,
    meta,
    eph_by_key,
    approx_time_gps,
    *,
    approx_llh,
    marker_llh=None,
    atmo_fn=None,
) -> dict:
    approx_rx = np.array(geometry.llh_to_ecef(*approx_llh))
    skipped = {}
    pr = {}
    sat_pos = {}
    per_sv = {}

    for sv in meta.get("provenance", {}).get("svs", []):
        key = f"{sv['sys']}{sv['prn']}"
        sysc = sv.get("sys")

        # a. Checks
        if sysc not in ("G", "J", "S", "E", "C", "R"):
            skipped[key] = "unsupported system"
            continue
        if sysc == "R" and sv.get("glo_k") is None:
            skipped[key] = "missing glo_k"
            continue
        if sv.get("band") not in iq_by_band:
            skipped[key] = "no band data"
            continue

        # b. Ephemeris lookup
        rinex_prn = sv["prn"] - engine._PRN_OFFSET.get(sysc, 0)
        rec = eph_by_key.get((sysc, rinex_prn))
        if rec is None and sysc == "G":
            rec = eph_by_key.get(rinex_prn)
        if rec is None:
            skipped[key] = "no ephemeris"
            continue

        # c. Signal, group delay, state function
        sig = signals.SIGNALS[
            {
                "G": "GPS_L1CA",
                "J": "QZSS_L1CA",
                "S": "SBAS_L1",
                "E": "GAL_E1",
                "C": "BDS_B1I",
                "R": "GLO_G1",
            }[sysc]
        ]
        svb = geometry.group_delay_bias_s(
            sysc, "L1", rec if isinstance(rec, dict) else None
        )
        state = geometry.state_fn_for(rec) if isinstance(rec, dict) else rec

        # d. Observables, elevation check, atmospheric delay, predicted phase
        o = geometry.observables(
            state, approx_rx, approx_time_gps, signal=sig, sv_clock_bias_s=svb
        )
        if o["el_deg"] < 5.0:
            skipped[key] = "low elevation"
            continue
        carrier_hz = sig.carrier_hz + (
            int(sv["glo_k"]) * 562_500.0 if sysc == "R" else 0.0
        )
        atmo_m = (
            float(
                atmo_fn(
                    math.radians(o["az_deg"]),
                    math.radians(o["el_deg"]),
                    carrier_hz,
                )
            )
            if atmo_fn
            else 0.0
        )
        pred_pr = o["pseudorange_m"] + atmo_m
        pred_phase = (pred_pr / config.C * sig.chip_rate_hz) % sig.code_len

        # e. Acquire L1 code phase
        iq, fs = iq_by_band[sv["band"]]
        acq = acquire_l1(
            iq,
            fs,
            sysc,
            sv["prn"],
            centre_offset_hz=sv["if_hz"],
            glo_k=(int(sv["glo_k"]) if sysc == "R" else None),
        )
        if acq["metric_db"] < 15.0:
            skipped[key] = f"weak ({acq['metric_db']:.1f} dB)"
            continue

        # f. Measured pseudorange
        err_chips = (
            (acq["code_phase_chips"] - pred_phase + sig.code_len / 2.0)
            % sig.code_len
        ) - sig.code_len / 2.0
        measured_pr = pred_pr + err_chips * (config.C / sig.chip_rate_hz)

        # g. Transmit time, corrected pseudorange
        pos, _vel, _tof, clk = geometry.solve_transmit_time(
            state, approx_rx, approx_time_gps
        )
        corrected_pr = measured_pr + config.C * (clk + svb) - atmo_m
        pr[key] = corrected_pr
        sat_pos[key] = pos
        per_sv[key] = {
            "measured_pr_m": measured_pr,
            "corrected_pr_m": corrected_pr,
            "metric_db": acq["metric_db"],
            "el_deg": o["el_deg"],
        }

    if len(pr) < 4:
        return {
            "error": f"only {len(pr)} SVs usable",
            "prns_used": sorted(pr),
            "skipped": skipped,
        }

    sol = solve_position(pr, sat_pos, x0=[*approx_rx, 0.0])
    lat, lon, h = _ecef_to_llh(*sol["ecef"])
    entries = []
    for k in sorted(pr):
        los = sat_pos[k] - np.array(sol["ecef"])
        entries.append({"_los": (los / np.linalg.norm(los)).tolist()})

    acquired: dict[str, list[int]] = {}
    for k in sorted(pr):
        s = k[0]
        p = int(k[1:])
        acquired.setdefault(s, []).append(p)
    for s in acquired:
        acquired[s].sort()

    out = {
        "ecef": sol["ecef"],
        "llh": [lat, lon, h],
        "clock_bias_s": sol["clock_bias_s"],
        "prns_used": sorted(pr),
        "pdop": geometry.dop(entries, sol["ecef"])["pdop"],
        "residual_rms_m": sol["residual_rms_m"],
        "per_sv": per_sv,
        "skipped": skipped,
        "acquired": acquired,
    }
    if marker_llh:
        truth = np.array(geometry.llh_to_ecef(*marker_llh))
        out["error_m"] = float(np.linalg.norm(np.array(sol["ecef"]) - truth))
    return out
