"""GLONASS k=7 teknik kanal desteği ve savunmacı sentez doğrulama testleri."""
import pytest
from backend.synth import signals, engine


def test_glo_channel_offset_accepts_k7():
    """GLONASS ICD Bölüm 3.1.1 uyarınca k=7 teknik kanalı kabul edilmeli."""
    assert signals.glo_channel_offset_hz(7, step_hz=562_500.0) == 7 * 562_500.0
    assert signals.glo_channel_offset_hz(7, step_hz=437_500.0) == 7 * 437_500.0


def test_glo_channel_offset_rejects_out_of_bounds():
    """k=-8 ve k=8 sınır dışı olup ValueError fırlatmalı."""
    with pytest.raises(ValueError):
        signals.glo_channel_offset_hz(8)
    with pytest.raises(ValueError):
        signals.glo_channel_offset_hz(-8)


def test_sv_spec_for_glonass_k7_success():
    """k=7 olan GLONASS uydusu _sv_spec_for tarafından başarıyla derlenmeli."""
    sig = signals.SIGNALS["GLO_G1"]
    entry = {
        "sys": "R",
        "prn": 28,
        "signal_id": sig,
        "glo_k": 7,
        "code_phase_chips": 0.0,
        "code_doppler_hz": 100.0,
        "carrier_freq_hz": 0.0,
        "carrier_doppler_hz": 200.0,
    }
    spec, keep = engine._sv_spec_for(entry, gain=1.0)
    assert keep is not None
    assert spec is not None
    # carrier_freq_hz: carrier_doppler_hz (200.0) + k * 562.5 kHz
    expected_freq = 200.0 + 7 * 562_500.0
    assert spec.carrier_freq_hz == pytest.approx(expected_freq)


def test_sv_spec_for_glonass_invalid_k_does_not_crash():
    """Bozuk/sınır dışı k değerine sahip uydu tüm simülasyonu çökertmemeli, atlanmalı."""
    sig = signals.SIGNALS["GLO_G1"]
    entry = {
        "sys": "R",
        "prn": 99,
        "signal_id": sig,
        "glo_k": 999,
        "code_phase_chips": 0.0,
        "code_doppler_hz": 100.0,
        "carrier_freq_hz": 0.0,
        "carrier_doppler_hz": 200.0,
    }
    spec, reason = engine._sv_spec_for(entry, gain=1.0)
    assert spec is None
    assert "geçersiz kanal" in reason
