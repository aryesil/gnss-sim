"""Engine seviyesinde regresyon: Klobuchar iyonosfer gecikmesi GPS sinyaline
enjekte edilir, Galileo (sys "E") sinyaline ENJEKTE EDİLMEZ.

Gerekçe (engine._atmo_for): I/NAV word 5 ai0..ai2 = 0 yayınlanıyor; NeQuick-G
alıcı bu durumda iyonosfer düzeltmesi çıkarmaz. E'ye Klobuchar gömmek alıcının
çıkaramayacağı bir yanlılık ve GPS ile tutarsız ISB üretip Galileo fix'ini
bozar. Test, klobuchar_delay_m çağrılarının (az, el) argümanlarını kaydeder:
Galileo uydularının bakış açısıyla hiç çağrı yapılmamalı, GPS için yapılmalı.
(Çağrı SAYISI karşılaştırılmaz: G-only ve G+E yolları aynı uydu için farklı
sayıda çağrı üretebilir; ayırt edici olan hangi bakış açılarının sorgulandığıdır.)
"""
import datetime as dt
import json
import math
import pathlib

import pytest

from backend import config
from backend.models import atmosphere
from backend.scenario import ScenarioRequest
from backend.synth import engine

_MIXED = str(pathlib.Path(__file__).parent.parent / "fixtures" / "brdc_mixed.rnx")
# test_native_e5a_closed_loop_fix.py ile aynı geometri: G1-G3 ve E3 yukarıda.
_RX = (-10.0, 180.0, 100.0)
_START = dt.datetime(2026, 8, 31, 0, 0, 22)


def _run(tmp_path, monkeypatch, systems):
    calls = []
    real = atmosphere.klobuchar_delay_m

    def spy(*a, **k):
        out = real(*a, **k)
        calls.append((out["delay_m"], a[5], a[6]))     # (gecikme, az_rad, el_rad)
        return out

    monkeypatch.setattr(config, "OUT_DIR", tmp_path)
    monkeypatch.setattr(engine.atmosphere, "klobuchar_delay_m", spy)
    req = ScenarioRequest(
        rinex_path=_MIXED, lat=_RX[0], lon=_RX[1], alt=_RX[2], start=_START,
        duration_s=1, sample_rate=2_600_000.0, sample_format="int8",
        engine="native", systems=systems, bands=["L1"], nav_message=False)
    outdir = engine.run(req)
    meta = json.loads((outdir / "meta.json").read_text())
    return calls, meta["provenance"]["svs"]


def _queried(calls, sv):
    az, el = math.radians(sv["az_deg"]), math.radians(sv["el_deg"])
    return any(abs(c[1] - az) < 0.01 and abs(c[2] - el) < 0.01 for c in calls)


def test_klobuchar_is_injected_for_gps_but_not_for_galileo(tmp_path, monkeypatch):
    calls, svs = _run(tmp_path, monkeypatch, ("G", "E"))
    gps = [sv for sv in svs if sv["sys"] == "G"]
    gal = [sv for sv in svs if sv["sys"] == "E"]
    assert gps and gal, "senaryoda hem GPS hem Galileo uydusu görünür olmalı"
    assert calls and all(c[0] > 0.0 for c in calls)
    # GPS: iyonosfer sorgulanır (G iono > 0)
    assert all(_queried(calls, sv) for sv in gps)
    # Galileo: iyonosfer hiç sorgulanmaz (E iono = 0)
    assert not any(_queried(calls, sv) for sv in gal)
