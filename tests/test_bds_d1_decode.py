"""Unit tests for BeiDou B1I D1 decoder (backend.analysis.bds_d1_decode)."""
import math
import numpy as np
import pytest

from backend.analysis import bds_d1_decode as D
from backend.analysis import bds_d1_encode as E
from backend.ephem import ephemeris

_WEEK, _SOW = 910, 561600.0


@pytest.fixture(scope="module")
def crec():
    e = ephemeris.parse_rinex_multi("tests/fixtures/brdc_mixed.rnx", ("C",),
                                    require=())
    rec = ephemeris.align_epochs(e, _WEEK, _SOW)
    rec = dict([v for k, v in rec.items() if k[0] == "C"][0])
    rec["prn"] = 6
    rec["toe"] = rec["toc"] = 561600.0 - 14.0 - (561600.0 - 14.0) % 8
    return rec


def test_bch_single_error_correction():
    rng = np.random.default_rng(42)
    info11 = rng.integers(0, 2, 11).tolist()
    cw = E.bch_encode(info11)

    dec_info, ok = D.bch_decode(cw)
    assert ok is True
    assert dec_info == info11

    # Corrupt a single bit at position 5
    corrupted = list(cw)
    corrupted[5] ^= 1
    dec_info_corr, ok_corr = D.bch_decode(corrupted)
    assert ok_corr is True
    assert dec_info_corr == info11


def test_decode_subframe_fields(crec):
    sow = 561594
    for k in (1, 2, 3):
        sf = E.d1_subframe(k, crec, sow + 6 * (k - 1), _WEEK)
        res = D.decode_subframe(sf)
        assert res["preamble_ok"] is True
        assert res["bch_ok"] is True
        assert res["fra_id"] == k
        assert res["sow"] == sow + 6 * (k - 1)
        fields = res["fields"]
        if k == 1:
            assert fields["week"] == _WEEK
            assert fields["toc"] == pytest.approx(crec["toc"], abs=1.0)
            assert fields["af0"] == pytest.approx(crec["af0"], abs=1e-10)
        elif k == 2:
            assert fields["m0"] == pytest.approx(crec["m0"], abs=1e-8)
            assert fields["e"] == pytest.approx(crec["e"], abs=1e-8)
            assert fields["sqrtA"] == pytest.approx(crec["sqrtA"], abs=1e-3)
        elif k == 3:
            assert fields["i0"] == pytest.approx(crec["i0"], abs=1e-8)
            assert fields["omega0"] == pytest.approx(crec["omega0"], abs=1e-8)


def test_decode_messages_from_stream(crec):
    stream, rate = E.nav_stream(crec, _WEEK, _SOW, 24.0)
    # Stream in {-1, +1} where bit 0 -> +1, bit 1 -> -1
    sym01 = [0 if s > 0 else 1 for s in stream]

    msgs = D.decode_messages(sym01)
    assert len(msgs) >= 3
    fra_ids = [m["fra_id"] for m in msgs]
    assert 1 in fra_ids and 2 in fra_ids and 3 in fra_ids


def test_decode_messages_offset_and_polarity_flip(crec):
    stream, rate = E.nav_stream(crec, _WEEK, _SOW, 24.0)
    sym01 = [0 if s > 0 else 1 for s in stream]

    # Add 13 offset symbols and invert polarity (0 <-> 1)
    shifted = [1] * 13 + [s ^ 1 for s in sym01]

    msgs = D.decode_messages(shifted)
    assert len(msgs) >= 3
    fra_ids = [m["fra_id"] for m in msgs]
    assert 1 in fra_ids and 2 in fra_ids and 3 in fra_ids


def test_reconstruct_ephemeris(crec):
    stream, rate = E.nav_stream(crec, _WEEK, _SOW, 24.0)
    sym01 = [0 if s > 0 else 1 for s in stream]
    msgs = D.decode_messages(sym01)

    rec = D.reconstruct_ephemeris(msgs, prn=6)
    assert rec["system"] == "C"
    assert rec["prn"] == 6
    assert rec["sqrtA"] == pytest.approx(crec["sqrtA"], abs=1e-3)
    assert rec["e"] == pytest.approx(crec["e"], abs=1e-8)
    assert rec["m0"] == pytest.approx(crec["m0"], abs=1e-8)
    assert rec["omega0"] == pytest.approx(crec["omega0"], abs=1e-8)
    assert rec["i0"] == pytest.approx(crec["i0"], abs=1e-8)
    assert rec["omega"] == pytest.approx(crec["omega"], abs=1e-8)
    assert rec["omega_dot"] == pytest.approx(crec["omega_dot"], abs=1e-10)
    assert rec["idot"] == pytest.approx(crec["idot"], abs=1e-10)
    assert rec["delta_n"] == pytest.approx(crec["delta_n"], abs=1e-10)
    assert rec["cuc"] == pytest.approx(crec["cuc"], abs=1e-7)
    assert rec["cus"] == pytest.approx(crec["cus"], abs=1e-7)
    assert rec["crc"] == pytest.approx(crec["crc"], abs=1.0)
    assert rec["crs"] == pytest.approx(crec["crs"], abs=1.0)
    assert rec["cic"] == pytest.approx(crec["cic"], abs=1e-7)
    assert rec["cis"] == pytest.approx(crec["cis"], abs=1e-7)
    assert rec["af0"] == pytest.approx(crec["af0"], abs=1e-9)
    assert rec["af1"] == pytest.approx(crec["af1"], abs=1e-12)
    assert rec["af2"] == pytest.approx(crec["af2"], abs=1e-15)
    assert rec["toe"] == pytest.approx(crec["toe"], abs=8.0)
    assert rec["toc"] == pytest.approx(crec["toc"], abs=8.0)
    assert rec["bdt_week"] == _WEEK


def test_demod_symbols_b1i():
    from backend.synth import _lib

    prn = 6
    code = _lib.code(3, prn, 2046, 0)[0].astype(np.float64)
    fs = 4_092_000.0
    ms_s = 0.001
    npp = int(round(fs * ms_s))
    nbits = 25
    nms = nbits * 20

    rng = np.random.default_rng(999)
    test_bits = rng.integers(0, 2, nbits)
    # bit 0 -> +1, bit 1 -> -1
    bit_syms = np.where(test_bits == 0, 1.0, -1.0)
    # Repeat each bit for 20 ms and multiply by Neuman-Hofman secondary code
    ms_syms = np.repeat(bit_syms, 20) * np.tile(D.NH20, nbits)

    dopp = 2500.0
    code_dopp = 2.046e6 / 1561.098e6 * dopp
    total_samples = nms * npp
    iq = _lib.debug_mix_range_ex(code.astype(np.int8), 2.046e6, 0.0, code_dopp, dopp, fs, 0, total_samples, sys=3, sub_hz=0.0)
    for k in range(nms):
        iq[k * npp:(k + 1) * npp] *= ms_syms[k]

    demod = D.demod_symbols_b1i(iq, fs, prn, dopp_hz=dopp, code_phase_chips=0.0)
    assert len(demod) == nbits
    match_direct = np.array_equal(demod, test_bits)
    match_inverted = np.array_equal(demod, 1 - test_bits)
    assert match_direct or match_inverted
