"""IONEX (IONosphere map EXchange format) parser, downloader, and interpolator.

Provides global Vertical Total Electron Content (VTEC) maps and converts VTEC
(TECU) to L1/L2/L5 ionospheric group delay (metres) for GNSS simulation and
SBAS Message Type 18 / 26 generation.
"""
from __future__ import annotations

import datetime as dt
import functools
import gzip
import logging
import math
import pathlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import requests

from backend import config

_log = logging.getLogger(__name__)

# 1 TECU = 10^16 e-/m^2
# Group delay (m) = 40.3 * TEC / f^2
# For GPS L1 (1575.42 MHz): 40.3 * 10^16 / (1575.42e6)^2 = 0.162371 m / TECU
TECU_TO_L1_DELAY_M = 40.3e16 / (config.L1_HZ ** 2)


@dataclass
class IonexMap:
    """Parsed IONEX global ionosphere dataset covering multiple epochs."""
    epochs_sow: np.ndarray        # 1D array of GPS seconds-of-week for each map
    lats: np.ndarray              # 1D array of latitudes (degrees, north to south)
    lons: np.ndarray              # 1D array of longitudes (degrees, west to east)
    tec: np.ndarray               # 3D array [n_epochs, n_lats, n_lons] in TECU
    height_km: float = 350.0      # Ionospheric shell height in km
    rms: np.ndarray | None = None # Optional RMS uncertainty array
    periodic_day: bool = False    # Sentetik 24 saatlik haritalarda haftalık SoW'u güne sarmak için

    def get_vtec(self, lat_deg: float, lon_deg: float, sow: float) -> float:
        """Interpolate Vertical TEC (TECU) at (lat, lon, sow)."""
        if self.periodic_day:
            # 24 saatlik sentetik IONEX haritaları için SoW gün içine sarılarak
            # multi-day simülasyonlarda son epoch'a kilitlenme engellenir.
            sow = float(sow) % 86400.0
        lat_clamped = max(float(self.lats[-1]), min(float(self.lats[0]), float(lat_deg)))
        # Normalize longitude into [-180, 180)
        lon_norm = ((float(lon_deg) + 180.0) % 360.0) - 180.0

        n_epochs, n_lats, n_lons = self.tec.shape

        # --- Time interpolation
        sows = self.epochs_sow
        if len(sows) > 1 and sow < sows[0] and sow + 604800.0 <= sows[-1]:
            sow += 604800.0
        if len(sows) == 1 or sow <= sows[0]:
            t_idx0, t_idx1, wt = 0, 0, 0.0
        elif sow >= sows[-1]:
            t_idx0, t_idx1, wt = n_epochs - 1, n_epochs - 1, 0.0
        else:
            t_idx1 = int(np.searchsorted(sows, sow))
            t_idx0 = max(0, t_idx1 - 1)
            dt_s = sows[t_idx1] - sows[t_idx0]
            wt = (sow - sows[t_idx0]) / dt_s if dt_s > 0 else 0.0

        # --- Spatial interpolation on grid
        # Latitude: self.lats is sorted descending (e.g. 87.5 to -87.5)
        # Find bracket:
        lat_step = abs(float(self.lats[1] - self.lats[0])) if n_lats > 1 else 1.0
        lat_top = float(self.lats[0])
        lat_frac = (lat_top - lat_clamped) / lat_step
        i0 = int(math.floor(lat_frac))
        i0 = max(0, min(n_lats - 2, i0))
        i1 = i0 + 1
        w_lat = max(0.0, min(1.0, lat_frac - i0))

        # Longitude: self.lons is ascending (-180.0 to 180.0)
        lon_step = abs(float(self.lons[1] - self.lons[0])) if n_lons > 1 else 1.0
        lon_left = float(self.lons[0])
        lon_frac = (lon_norm - lon_left) / lon_step
        j0 = int(math.floor(lon_frac))
        j0 = max(0, min(n_lons - 1, j0))
        j1 = (j0 + 1) % n_lons
        w_lon = max(0.0, min(1.0, lon_frac - j0))

        def interp_2d(t_idx: int) -> float:
            grid = self.tec[t_idx]
            v00 = float(grid[i0, j0])
            v01 = float(grid[i0, j1])
            v10 = float(grid[i1, j0])
            v11 = float(grid[i1, j1])
            top = v00 * (1.0 - w_lon) + v01 * w_lon
            bot = v10 * (1.0 - w_lon) + v11 * w_lon
            return top * (1.0 - w_lat) + bot * w_lat

        v0 = interp_2d(t_idx0)
        if wt == 0.0 or t_idx0 == t_idx1:
            return max(0.0, v0)
        v1 = interp_2d(t_idx1)
        return max(0.0, v0 * (1.0 - wt) + v1 * wt)

    def get_vertical_delay_m(self, lat_deg: float, lon_deg: float, sow: float,
                             freq_hz: float = config.L1_HZ) -> float:
        """Convert vertical TEC to ionospheric path delay in metres at freq_hz."""
        vtec = self.get_vtec(lat_deg, lon_deg, sow)
        scale = 40.3e16 / (freq_hz ** 2)
        return vtec * scale


def parse_ionex(content: str | bytes | pathlib.Path) -> IonexMap:
    """Parse an IONEX 1.0 text or gzip/compressed file into an IonexMap."""
    if isinstance(content, (pathlib.Path, str)):
        p = pathlib.Path(content)
        if p.is_file():
            raw = p.read_bytes()
            if p.suffix in (".gz", ".GZ") or raw[:2] == b"\x1f\x8b":
                text = gzip.decompress(raw).decode("latin-1", errors="replace")
            else:
                text = raw.decode("latin-1", errors="replace")
        else:
            text = str(content)
    elif isinstance(content, bytes):
        if content[:2] == b"\x1f\x8b":
            text = gzip.decompress(content).decode("latin-1", errors="replace")
        else:
            text = content.decode("latin-1", errors="replace")
    else:
        raise TypeError("content must be str, bytes, or Path")

    lines = text.splitlines()

    exponent = -1
    hgt_km = 350.0
    lat1, lat2, dlat = 87.5, -87.5, -2.5
    lon1, lon2, dlon = -180.0, 180.0, 5.0

    in_header = True
    header_end_idx = 0
    for idx, line in enumerate(lines):
        tag = line[60:].strip() if len(line) >= 60 else ""
        if tag == "EXPONENT":
            try:
                exponent = int(line[:60].strip())
            except ValueError:
                pass
        elif tag == "HGT1 / HGT2 / DHGT":
            parts = line[:60].split()
            if parts:
                try:
                    hgt_km = float(parts[0])
                except ValueError:
                    pass
        elif tag == "LAT1 / LAT2 / DLAT":
            parts = line[:60].split()
            if len(parts) >= 3:
                try:
                    lat1, lat2, dlat = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    pass
        elif tag == "LON1 / LON2 / DLON":
            parts = line[:60].split()
            if len(parts) >= 3:
                try:
                    lon1, lon2, dlon = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    pass
        elif tag == "END OF HEADER":
            in_header = False
            header_end_idx = idx + 1
            break

    # Build lat and lon coordinates
    if dlat == 0:
        dlat = -2.5
    if dlon == 0:
        dlon = 5.0
    n_lat = int(round(abs(lat1 - lat2) / abs(dlat))) + 1
    lats = np.linspace(lat1, lat2, n_lat)
    n_lon = int(round(abs(lon2 - lon1) / abs(dlon))) + 1
    lons = np.linspace(lon1, lon2, n_lon)

    scale = 10.0 ** exponent

    maps: list[np.ndarray] = []
    epochs_sow: list[float] = []

    cur_map = None
    cur_lat = None
    cur_row_vals: list[int] = []
    in_tec = False

    for line in lines[header_end_idx:]:
        if "START OF TEC MAP" in line:
            in_tec = True
            cur_map = np.zeros((n_lat, n_lon), dtype=np.float32)
            cur_lat = None
            cur_row_vals = []
            continue
        if "END OF TEC MAP" in line:
            if cur_lat is not None and cur_row_vals and cur_map is not None:
                lat_idx = int(round((lat1 - cur_lat) / abs(dlat)))
                if 0 <= lat_idx < n_lat:
                    nv = min(n_lon, len(cur_row_vals))
                    cur_map[lat_idx, :nv] = np.array(cur_row_vals[:nv], dtype=np.float32) * scale
                cur_row_vals = []
            if cur_map is not None:
                maps.append(cur_map)
            in_tec = False
            cur_map = None
            cur_lat = None
            continue
        if not in_tec:
            continue

        if "EPOCH OF CURRENT MAP" in line:
            parts = [int(p) for p in line[:60].split()[:6]]
            if len(parts) == 6:
                yr, mo, dy, hr, mn, sc = parts
                epoch_dt = dt.datetime(yr, mo, dy, hr, mn, sc, tzinfo=dt.timezone.utc)
                gps_t0 = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
                tot_s = (epoch_dt - gps_t0).total_seconds()
                sow = tot_s % 604800.0
                epochs_sow.append(sow)
            continue

        if "LAT/LON1/LON2/DLON/H" in line:
            if cur_lat is not None and cur_row_vals and cur_map is not None:
                lat_idx = int(round((lat1 - cur_lat) / abs(dlat)))
                if 0 <= lat_idx < n_lat:
                    nv = min(n_lon, len(cur_row_vals))
                    arr = np.array(cur_row_vals[:nv], dtype=np.float32)
                    arr[arr == 9999.0] = 0.0
                    cur_map[lat_idx, :nv] = arr * scale
                cur_row_vals = []
            try:
                # First token or fixed columns 2..8 contains the latitude
                lat_str = line[:8].strip()
                cur_lat = float(lat_str) if lat_str else float(line[:60].split()[0])
            except ValueError:
                cur_lat = None
            continue

        # Data rows
        if cur_map is not None and cur_lat is not None:
            for token in line.split():
                try:
                    v = int(token)
                    cur_row_vals.append(0 if v == 9999 else v)
                except ValueError:
                    pass

    # Flush last row of last map if any
    if cur_lat is not None and cur_row_vals and cur_map is not None:
        lat_idx = int(round((lat1 - cur_lat) / abs(dlat)))
        if 0 <= lat_idx < n_lat:
            nv = min(n_lon, len(cur_row_vals))
            arr = np.array(cur_row_vals[:nv], dtype=np.float32)
            arr[arr == 9999.0] = 0.0
            cur_map[lat_idx, :nv] = arr * scale

    if not maps:
        # Fallback to single blank or synthetic map
        maps = [np.zeros((n_lat, n_lon), dtype=np.float32)]
        epochs_sow = [0.0]

    tec_array = np.stack(maps, axis=0)
    sows_arr = np.asarray(epochs_sow, dtype=np.float64)
    if len(sows_arr) > 1 and sows_arr[-1] < sows_arr[0]:
        for idx in range(1, len(sows_arr)):
            if sows_arr[idx] < sows_arr[idx - 1]:
                sows_arr[idx:] += 604800.0
    return IonexMap(
        epochs_sow=sows_arr,
        lats=lats,
        lons=lons,
        tec=tec_array,
        height_km=hgt_km,
    )


@functools.lru_cache(maxsize=16)
def synthetic_ionex(date: dt.date | dt.datetime | None = None,
                    alpha: tuple | None = None,
                    beta: tuple | None = None) -> IonexMap:
    """Generate a realistic global VTEC map across 24 hours using Klobuchar.
    Ensures simulation functions reliably even when offline without network access.
    """
    from backend.models import atmosphere
    a = alpha or atmosphere.DEFAULT_KLOBUCHAR_ALPHA
    b = beta or atmosphere.DEFAULT_KLOBUCHAR_BETA

    n_lat = 71  # 87.5 to -87.5 step -2.5
    n_lon = 73  # -180.0 to 180.0 step 5.0
    lats = np.linspace(87.5, -87.5, n_lat)
    lons = np.linspace(-180.0, 180.0, n_lon)

    # 13 epochs across the day (every 2 hours, 0..86400 s)
    sows = np.linspace(0.0, 86400.0, 13)
    tec_maps = np.zeros((13, n_lat, n_lon), dtype=np.float32)

    for t_idx, sow in enumerate(sows):
        for i, lat in enumerate(lats):
            lat_rad = math.radians(lat)
            for j, lon in enumerate(lons):
                lon_rad = math.radians(lon)
                # Compute zenith delay (el = 90 deg)
                res = atmosphere.klobuchar_delay_m(
                    a, b, sow, lat_rad, lon_rad, 0.0, math.pi / 2.0
                )
                vert_delay_m = res.get("vertical_delay_m", 2.0)
                # Convert delay (m) back to TECU
                tecu = vert_delay_m / TECU_TO_L1_DELAY_M
                tec_maps[t_idx, i, j] = tecu

    return IonexMap(
        epochs_sow=sows,
        lats=lats,
        lons=lons,
        tec=tec_maps,
        height_km=350.0,
        periodic_day=True,
    )


def download_ionex(target_date: dt.date | dt.datetime,
                   cache_dir: pathlib.Path | None = None,
                   mirrors: Sequence[str] | None = None) -> pathlib.Path | None:
    """Download daily IGS IONEX global TEC map for target_date.
    Returns local cached Path if successful, or None on failure.
    """
    cdir = cache_dir or getattr(config, "IONEX_DIR", config.DATA_DIR / "ionex")
    cdir.mkdir(parents=True, exist_ok=True)

    if isinstance(target_date, dt.datetime):
        target_date = target_date.date()

    yyyy = f"{target_date.year:04d}"
    yy = f"{target_date.year % 100:02d}"
    doy = f"{target_date.timetuple().tm_yday:03d}"

    # Compute GPS week
    epoch_dt = dt.datetime(target_date.year, target_date.month, target_date.day,
                           tzinfo=dt.timezone.utc)
    gps_t0 = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
    gpsweek = str(int((epoch_dt - gps_t0).total_seconds() // 604800))

    dest_gz = cdir / f"ESA0OPSRAP_{yyyy}{doy}0000_01D_02H_GIM.INX.gz"
    if dest_gz.is_file() and dest_gz.stat().st_size > 1024:
        return dest_gz

    m_list = list(mirrors) if mirrors else getattr(config, "IONEX_MIRRORS", [
        "http://navigation-office.esa.int/products/gnss-products/{gpsweek}/ESA0OPSRAP_{yyyy}{doy}0000_01D_02H_GIM.INX.gz",
        "http://navigation-office.esa.int/products/gnss-products/{gpsweek}/ESA0OPSFIN_{yyyy}{doy}0000_01D_02H_GIM.INX.gz",
    ])

    for tmpl in m_list:
        url = tmpl.format(yyyy=yyyy, yy=yy, doy=doy, gpsweek=gpsweek)
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 1024:
                # Verify it's actually an IONEX file
                try:
                    decomp = gzip.decompress(resp.content)
                    if b"IONEX VERSION" in decomp[:2048]:
                        dest_gz.write_bytes(resp.content)
                        _log.info("Downloaded IONEX map to %s from %s", dest_gz, url)
                        return dest_gz
                except Exception:
                    if b"IONEX VERSION" in resp.content[:1024]:
                        dest_raw = cdir / f"ESA0OPSRAP_{yyyy}{doy}0000_01D_02H_GIM.INX"
                        dest_raw.write_bytes(resp.content)
                        return dest_raw
        except Exception as exc:
            _log.debug("IONEX download failed for %s: %s", url, exc)

    return None


@functools.lru_cache(maxsize=16)
def load_or_fetch_ionex(target_date: dt.date | dt.datetime | None = None,
                        cache_dir: pathlib.Path | None = None) -> IonexMap:
    """Retrieve IonexMap for target_date (downloading if needed, or synthetic fallback)."""
    d = target_date or dt.date.today()
    path = download_ionex(d, cache_dir=cache_dir)
    if path and path.is_file():
        try:
            return parse_ionex(path)
        except Exception as exc:
            _log.warning("Failed parsing downloaded IONEX %s: %s; falling back to synthetic", path, exc)
    return synthetic_ionex(d)
