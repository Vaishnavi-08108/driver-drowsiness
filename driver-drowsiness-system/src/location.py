"""
location.py
Location tracking and geolocation provider for the Driver Drowsiness Detection System.

Priority for locating the driver during an emergency:
    1. Manual coordinates (if set by the user).
    2. Fresh browser location (from HTML5 Geolocation API).
    3. IP-based location fallback (via ipapi.co).
    4. Stale browser location (if IP lookup fails).
"""

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

from src import config

SOURCE_BROWSER = "browser"
SOURCE_IP = "ip"
SOURCE_MANUAL = "manual"

BROWSER_MAX_AGE_SEC = 30.0
BROWSER_MAX_ACCURACY_M = 100.0
IP_CACHE_SEC = 1800.0
IP_RETRY_SEC = 60.0


@dataclass(frozen=True)
class Location:
    """Represents a geographic position."""

    lat: float
    lon: float
    source: str
    place: str = ""
    accuracy_m: Optional[float] = None
    timestamp: float = 0.0


def valid_coordinates(lat: Any, lon: Any) -> bool:
    """Validate that lat and lon are valid, non-zero geographic coordinates."""
    if lat is None or lon is None:
        return False
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (ValueError, TypeError):
        return False

    if math.isnan(lat_f) or math.isnan(lon_f) or math.isinf(lat_f) or math.isinf(lon_f):
        return False

    # (0, 0) is almost certainly invalid/placeholder data
    if abs(lat_f) < 1e-6 and abs(lon_f) < 1e-6:
        return False

    return -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0


def make_location(
    lat: Any,
    lon: Any,
    source: str,
    place: str = "",
    accuracy_m: Optional[Any] = None,
    now: Optional[float] = None,
) -> Location:
    """Construct a validated Location instance."""
    lat_f = float(lat)
    lon_f = float(lon)
    t = time.time() if now is None else float(now)

    acc = None
    if accuracy_m is not None:
        try:
            val = float(accuracy_m)
            if val > 0 and math.isfinite(val):
                acc = val
        except (ValueError, TypeError):
            acc = None

    return Location(
        lat=lat_f,
        lon=lon_f,
        source=str(source),
        place=str(place),
        accuracy_m=acc,
        timestamp=t,
    )


def map_link(location: Location) -> str:
    """Generate a Google Maps URL for the given location."""
    return f"https://www.google.com/maps?q={location.lat:.5f},{location.lon:.5f}"


def describe(location: Location) -> str:
    """Generate a human-readable description of the location."""
    parts = []
    if location.source == SOURCE_MANUAL:
        parts.append(f"{location.lat:.4f}, {location.lon:.4f} (entered by hand)")
    elif location.source == SOURCE_IP:
        desc = f"{location.lat:.4f}, {location.lon:.4f} (approximate, via IP address"
        if location.place:
            desc += f": {location.place}"
        desc += ")"
        parts.append(desc)
    else:
        acc_text = f" (~{int(round(location.accuracy_m))} m accuracy)" if location.accuracy_m is not None else ""
        parts.append(f"{location.lat:.4f}, {location.lon:.4f}{acc_text}")

    return " ".join(parts)


def parse_browser_location(payload: Any, now: Optional[float] = None) -> Optional[Location]:
    """Parse geolocation JSON payload received from browser JavaScript."""
    if not isinstance(payload, dict) or "error" in payload:
        return None

    if "coords" in payload and isinstance(payload["coords"], dict):
        coords = payload["coords"]
        lat = coords.get("latitude")
        lon = coords.get("longitude")
        acc = coords.get("accuracy")
    else:
        lat = payload.get("latitude")
        lon = payload.get("longitude")
        acc = payload.get("accuracy")

    if not valid_coordinates(lat, lon):
        return None

    received = time.time() if now is None else float(now)
    try:
        captured = float(payload.get("timestamp", received * 1000)) / 1000
    except (TypeError, ValueError):
        return None
    if not math.isfinite(captured) or not 0 <= received - captured <= BROWSER_MAX_AGE_SEC:
        return None
    return make_location(lat, lon, SOURCE_BROWSER, accuracy_m=acc, now=captured)


def fetch_ip_location(
    opener: Callable[..., Any] = urllib.request.urlopen,
    url: str = config.LOCATION_IP_URL,
    timeout: float = config.LOCATION_TIMEOUT_SEC,
    now: Optional[float] = None,
) -> Optional[Location]:
    """Fetch geographic location based on public IP address."""
    if not url.startswith("https://ipapi.co"):
        return None

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "driver-drowsiness-system/1.0"})
        with opener(req, timeout=timeout) as resp:
            raw = resp.read(65536)
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict) or data.get("error"):
                return None

            lat = data.get("latitude")
            lon = data.get("longitude")
            if not valid_coordinates(lat, lon):
                return None

            place_parts = [data.get("city"), data.get("region"), data.get("country_name")]
            place = ", ".join(p for p in place_parts if p)

            return make_location(lat, lon, SOURCE_IP, place=place, now=now)
    except Exception:
        return None


class LocationProvider:
    """Manages location resolution across browser GPS, manual override, and IP fallbacks."""

    def __init__(self, ip_fetcher: Optional[Callable[..., Optional[Location]]] = None) -> None:
        self._fetcher = ip_fetcher if ip_fetcher is not None else fetch_ip_location
        self._manual: Optional[Location] = None
        self._browser: Optional[Location] = None
        self._ip: Optional[Location] = None
        self._last_ip_attempt: float = -1e9

    def update_from_browser(self, payload: Any, now: Optional[float] = None) -> bool:
        """Update provider with new browser geolocation payload."""
        loc = parse_browser_location(payload, now=now)
        if loc is not None:
            self._browser = loc
            return True
        self._browser = None
        return False

    def set_manual(self, lat: Any, lon: Any, now: Optional[float] = None) -> bool:
        """Set a manual coordinate override."""
        if not valid_coordinates(lat, lon):
            return False
        self._manual = make_location(lat, lon, SOURCE_MANUAL, now=now)
        return True

    def clear_manual(self) -> None:
        """Clear any active manual coordinate override."""
        self._manual = None

    def current_precise(self, now: Optional[float] = None) -> Optional[Location]:
        """Resolve only recent, sufficiently accurate device coordinates for alerts."""
        t = time.time() if now is None else float(now)

        if (self._browser is not None
                and 0 <= t - self._browser.timestamp <= BROWSER_MAX_AGE_SEC
                and self._browser.accuracy_m is not None
                and self._browser.accuracy_m <= BROWSER_MAX_ACCURACY_M):
            return self._browser

        return None

    def current(self, now: Optional[float] = None) -> Optional[Location]:
        """Resolve current location according to priority rules."""
        t = time.time() if now is None else float(now)

        # 1. Manual coordinates override everything
        if self._manual is not None:
            return self._manual

        # 2. Fresh browser location
        if self._browser is not None and (t - self._browser.timestamp <= BROWSER_MAX_AGE_SEC):
            return self._browser

        # 3. IP location cache check
        if self._ip is not None and (t - self._ip.timestamp <= IP_CACHE_SEC):
            return self._ip

        # 4. Attempt IP fetch if throttled retry window has passed
        if (t - self._last_ip_attempt) >= IP_RETRY_SEC:
            self._last_ip_attempt = t
            try:
                new_ip = self._fetcher(now=t)
                if new_ip is not None:
                    self._ip = new_ip
                    return self._ip
            except Exception:
                pass

        # If IP fetch failed, an expired IP result survives
        if self._ip is not None:
            return self._ip

        # If no IP, stale browser location is the next fallback
        if self._browser is not None:
            return self._browser

        return None
