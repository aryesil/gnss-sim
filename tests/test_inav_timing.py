"""Timing verification tests for Galileo E1 I/NAV navigation stream (Stage 4).

Verifies:
1. inav_decode.decode_messages annotates each decoded page with sym_idx.
2. The transmit time t_tx = sow + sym_idx / 250.0 matches the TOW broadcast in Word Types 5 and 0.
3. No 4 ms, 1 s, or 2 s offsets exist between transmit time and broadcast TOW.
"""
import pytest
from backend.analysis import inav_decode, inav_encode
from backend.ephem import ephemeris


@pytest.fixture(scope="module")
def galileo_eph():
    ephs = ephemeris.parse_rinex_multi("tests/fixtures/brdc_mixed.rnx", ("E",), require=())
    rec = [v for k, v in ephs.items() if k[0] == "E"][0]
    return rec


def test_inav_page_timing_and_tow_alignment(galileo_eph):
    """Galileo I/NAV 2 saniye nominal sayfa iletim zamanı (t_tx) ile
    Word 5 ve Word 0 içinde yayınlanan TOW arasındaki bağıntıyı doğrula.
    Galileo OS SIS ICD Madde 4.1.3 gereği TOW, sayfanın başlangıç anına (t_tx) eşittir.
    """
    sow = 100000.0
    week = 1410
    duration_s = 120.0

    arr = inav_encode.nav_stream(galileo_eph, week, sow, duration_s=duration_s)
    msgs = inav_decode.decode_messages(arr)
    assert len(msgs) >= 30, f"En az 30 sayfa çözülmeli, bulunan: {len(msgs)}"

    tow_checked = 0
    for idx, page in enumerate(msgs):
        assert "sym_idx" in page, f"Sayfa {idx} 'sym_idx' alanını taşımalı"
        sym_idx = page["sym_idx"]
        t_tx = sow + sym_idx / 250.0

        wtype = page["type"]
        tow = page["fields"].get("tow")
        if tow is not None:
            # Word 5 ve Word 0'da TOW yayınlanır.
            assert wtype in (0, 5)
            # t_tx ile tow birebir eşit olmalı (sıfır kayma)
            assert t_tx == pytest.approx(tow, abs=1e-6), (
                f"Zamanlama kayması tespit edildi! Sayfa {idx} (tip {wtype}): "
                f"t_tx={t_tx}, tow={tow}, fark={t_tx - tow} s"
            )
            tow_checked += 1

    assert tow_checked >= 10, f"En az 10 adet TOW içeren sayfa test edilmeli, kontrol edilen: {tow_checked}"
