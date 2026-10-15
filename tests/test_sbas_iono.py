"""Tests for DO-229 SBAS Message Types 18 & 26 (Ionosphere) and IONEX processing."""
import numpy as np
import pytest

from backend.analysis import sbas_decode, sbas_iono
from backend.ephem import ephemeris, ionex


@pytest.fixture(scope="module")
def srec():
    e = ephemeris.parse_rinex_multi("tests/fixtures/brdc_mixed.rnx", ("S",), require=())
    rec = [v for k, v in e.items() if k[0] == "S"][0]
    rec["prn"] = 120
    return rec


def test_band_tables_geometry():
    """Verify that all 11 DO-229 bands are correctly structured."""
    for b in range(11):
        igps = sbas_iono.get_band_igps(b)
        assert len(igps) in (192, 200, 201)
        for lat, lon in igps:
            assert -90.0 <= lat <= 90.0
            assert -180.0 <= lon <= 180.0

    # Test band localization
    assert sbas_iono.find_best_band(41.0, 29.0) in (4, 5)   # Istanbul/Europe
    assert sbas_iono.find_best_band(37.0, -122.0) == 1       # California
    assert sbas_iono.find_best_band(70.0, 10.0) == 9         # North polar
    assert sbas_iono.find_best_band(-70.0, 10.0) == 10       # South polar


def test_type18_encode_and_decode_roundtrip():
    """Verify Message Type 18 250-bit block encoding and decoding."""
    band = 5
    mask = [0] * 201
    for active_idx in (10, 25, 40, 100, 150):
        mask[active_idx] = 1

    blk = sbas_iono.build_type18_block(idx=1, band=band, mask_bits=mask, iodi=2, num_bands=1)
    assert len(blk) == 250

    decoded = sbas_decode.decode_block(blk)
    assert decoded is not None
    assert decoded["crc_ok"] is True
    assert decoded["type"] == 18
    assert decoded["band"] == band
    assert decoded["iodi"] == 2
    assert decoded["num_bands"] == 1
    assert decoded["active_igps"] == [10, 25, 40, 100, 150]


def test_type26_encode_and_decode_roundtrip():
    """Verify Message Type 26 250-bit block encoding and decoding."""
    band = 4
    block_id = 0
    delays = [1.25, 2.50, 3.75, 5.0, 0.125, 8.0, 10.5, 4.25, 2.0, 1.0, 0.5, 6.0, 7.5, 3.0, 2.25]
    giveis = [2, 3, 4, 2, 1, 3, 5, 2, 3, 2, 1, 4, 3, 2, 3]

    blk = sbas_iono.build_type26_block(idx=2, band=band, block_id=block_id,
                                       delays_m=delays, givei_list=giveis, iodi=1)
    assert len(blk) == 250

    decoded = sbas_decode.decode_block(blk)
    assert decoded is not None
    assert decoded["crc_ok"] is True
    assert decoded["type"] == 26
    assert decoded["band"] == band
    assert decoded["block_id"] == block_id
    assert decoded["iodi"] == 1
    assert len(decoded["igps"]) == 15

    for orig_d, orig_g, dec_igp in zip(delays, giveis, decoded["igps"]):
        assert dec_igp["delay_m"] == pytest.approx(orig_d, abs=0.0625)
        assert dec_igp["givei"] == orig_g


def test_ionex_synthetic_and_interpolation():
    """Verify synthetic VTEC generation and spatial/temporal interpolation."""
    imap = ionex.synthetic_ionex()
    assert imap.tec.shape == (13, 71, 73)
    assert imap.tec.min() >= 0.0

    # Midday delay at Istanbul should be positive and reasonable
    midday_delay = imap.get_vertical_delay_m(41.0, 29.0, 43200.0)
    assert 1.0 < midday_delay < 15.0

    # Nighttime delay should be significantly lower than midday delay
    night_delay = imap.get_vertical_delay_m(41.0, 29.0, 0.0)
    assert 0.2 <= night_delay < midday_delay


def test_type1_encode_and_decode_roundtrip():
    """Verify Message Type 1 (PRN Mask) 250-bit block encoding and decoding."""
    gps_prns = [1, 3, 5, 14, 21, 32]
    blk = sbas_iono.build_type1_block(idx=1, gps_prns=gps_prns, iodp=2)
    assert len(blk) == 250

    decoded = sbas_decode.decode_block(blk)
    assert decoded is not None
    assert decoded["crc_ok"] is True
    assert decoded["type"] == 1
    assert decoded["iodp"] == 2
    assert decoded["active_gps_prns"] == gps_prns


def test_type2_encode_and_decode_roundtrip():
    """Verify Message Type 2 (Fast Corrections) 250-bit block encoding and decoding."""
    prcs = [0.125 * i for i in range(-6, 7)]  # 13 PRCs from -0.75m to +0.75m
    udres = [3] * 13
    blk = sbas_iono.build_type2_block(idx=2, prc_list=prcs, udre_list=udres, iodf=1, iodp=2)
    assert len(blk) == 250

    decoded = sbas_decode.decode_block(blk)
    assert decoded is not None
    assert decoded["crc_ok"] is True
    assert decoded["type"] == 2
    assert decoded["iodf"] == 1
    assert decoded["iodp"] == 2
    assert len(decoded["prc_m"]) == 13
    for orig, dec in zip(prcs, decoded["prc_m"]):
        assert dec == pytest.approx(orig, abs=0.01)
    assert decoded["udre"] == udres


def test_type25_encode_and_decode_roundtrip():
    """Verify Message Type 25 (Long-Term Satellite Error Corrections) 250-bit block encoding and decoding."""
    sats = [
        {"prn_mask_idx": 1, "iode": 42, "dx_m": 0.5, "dy_m": -1.25, "dz_m": 2.125, "daf0_s": 1e-7},
        {"prn_mask_idx": 2, "iode": 105, "dx_m": 0.0, "dy_m": 0.0, "dz_m": 0.0, "daf0_s": 0.0},
        {"prn_mask_idx": 3, "iode": 200, "dx_m": -3.0, "dy_m": 4.5, "dz_m": -0.25, "daf0_s": -2e-7},
    ]
    blk = sbas_iono.build_type25_block(idx=0, sat_corrections=sats, iodp=1)
    assert len(blk) == 250

    decoded = sbas_decode.decode_block(blk)
    assert decoded is not None
    assert decoded["crc_ok"] is True
    assert decoded["type"] == 25
    assert decoded["iodp"] == 1
    assert len(decoded["sats"]) == 3

    dec_sats = decoded["sats"]
    assert dec_sats[0]["prn_mask_idx"] == 1
    assert dec_sats[0]["iode"] == 42
    assert dec_sats[0]["dx_m"] == pytest.approx(0.5, abs=0.0625)
    assert dec_sats[0]["dy_m"] == pytest.approx(-1.25, abs=0.0625)
    assert dec_sats[0]["dz_m"] == pytest.approx(2.125, abs=0.0625)

    assert dec_sats[1]["prn_mask_idx"] == 2
    assert dec_sats[1]["iode"] == 105
    assert dec_sats[1]["dx_m"] == 0.0

    assert dec_sats[2]["prn_mask_idx"] == 3
    assert dec_sats[2]["iode"] == 200


def test_type7_encode_and_decode_roundtrip():
    """Verify Message Type 7 (Fast Correction Degradation Factor) 250-bit block encoding and decoding."""
    ai = [1] * 12 + [2] * 4
    blk = sbas_iono.build_type7_block(idx=1, ai_list=ai, t_lat=1, iodp=2)
    assert len(blk) == 250

    decoded = sbas_decode.decode_block(blk)
    assert decoded is not None
    assert decoded["crc_ok"] is True
    assert decoded["type"] == 7
    assert decoded["t_lat"] == 1
    assert decoded["iodp"] == 2
    assert decoded["ai_list"][:16] == ai
    assert decoded["ai_list"][16:] == [0] * (51 - 16)


def test_sbas_nav_stream_generates_mt1_mt2_mt7_mt9_mt18_mt25_mt26(srec):
    """Verify that nav_stream produces a clean 500 sym/s stream containing Types 1, 2, 7, 9, 18, 25, and 26."""
    arr, rate = sbas_iono.nav_stream(srec, 0, 561600.0, 12, rx_lat=41.0, rx_lon=29.0,
                                     gps_prns=[1, 3, 5, 7, 8, 9, 11, 14, 17, 19, 21, 28, 30])
    assert rate == 500.0
    assert len(arr) == int(12 + 30) * 500

    blks = sbas_decode.decode_messages(arr)
    assert len(blks) >= 12
    assert all(b["crc_ok"] for b in blks)

    types = [b["type"] for b in blks]
    assert 1 in types, "Stream must contain Type 1 (PRN Mask)"
    assert 2 in types, "Stream must contain Type 2 (Fast Corrections)"
    assert 7 in types, "Stream must contain Type 7 (Fast Correction Degradation Factor)"
    assert 9 in types, "Stream must contain Type 9 (GEO Navigation Message)"
    assert 18 in types, "Stream must contain Type 18 (Ionospheric Grid Point Mask)"
    assert 25 in types, "Stream must contain Type 25 (Long-Term Satellite Error Corrections)"
    assert 26 in types, "Stream must contain Type 26 (Ionospheric Delay Corrections)"


def test_sbas_nav_stream_is_deterministic(srec):
    """Verify deterministic symbol generation."""
    a, _ = sbas_iono.nav_stream(srec, 0, 561600.0, 6, rx_lat=41.0, rx_lon=29.0)
    b, _ = sbas_iono.nav_stream(srec, 0, 561600.0, 6, rx_lat=41.0, rx_lon=29.0)
    assert np.array_equal(a, b)
    assert sorted(set(a.tolist())) == [-1, 1]

