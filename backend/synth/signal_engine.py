from __future__ import annotations

import pathlib

from backend import generator, scenario


def run(req: scenario.ScenarioRequest, progress_cb=None,
        binary: str | None = None) -> pathlib.Path:
    """Tüm sinyal üretimi C++ native motor (backend.synth.engine) üzerinden yürütülür.
    gps-sdr-sim emekliye ayrılmıştır; geriye dönük çağrılar da native motora iletilir.
    """
    if req.engine not in ("native", "gps-sdr-sim"):
        raise ValueError(f"unknown engine {req.engine!r} (expected 'native')")
    from backend.synth import engine as native_engine
    return native_engine.run(req, progress_cb=progress_cb)
