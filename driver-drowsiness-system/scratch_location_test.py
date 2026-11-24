import json
import urllib.error

from src import config
from src.location import (
    BROWSER_MAX_AGE_SEC, IP_CACHE_SEC, IP_RETRY_SEC,
    SOURCE_BROWSER, SOURCE_IP, SOURCE_MANUAL,
    LocationProvider, describe, fetch_ip_location, make_location,
    map_link, parse_browser_location, valid_coordinates,
)

BLR = (12.9716, 77.5946)


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self, n=-1):
        return self._body[:n] if n > 0 else self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def opener_returning(payload):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return lambda request, timeout=None: FakeResponse(body)


def failing_opener(request, timeout=None):
    raise urllib.error.URLError("no internet")


class CountingFetcher:
    """Stands in for fetch_ip_location and counts the calls."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def __call__(self, now=None):
        self.calls += 1
        return self.results.pop(0) if self.results else None


# 1. Coordinate validation
assert valid_coordinates(12.9, 77.5) and valid_coordinates("12.9", "77.5")
assert not valid_coordinates(91, 0.5) and not valid_coordinates(0.5, 181)
assert not valid_coordinates(0, 0)                      # almost certainly "no data"
assert not valid_coordinates(float("nan"), 5) and not valid_coordinates(None, 5)
assert not valid_coordinates("abc", 5)
assert make_location(*BLR, SOURCE_MANUAL, accuracy_m=-5, now=1.0).accuracy_m is None
assert make_location(*BLR, SOURCE_MANUAL, accuracy_m=25, now=1.0).accuracy_m == 25.0

# 2. Map link and description
loc = make_location(*BLR, SOURCE_BROWSER, accuracy_m=18.4, now=1.0)
assert map_link(loc) == "https://www.google.com/maps?q=12.97160,77.59460"
assert "18 m" in describe(loc)
assert "IP address" in describe(make_location(*BLR, SOURCE_IP, place="Bengaluru", now=1.0))
assert "by hand" in describe(make_location(*BLR, SOURCE_MANUAL, now=1.0))

# 3. Browser payloads
nested = {"coords": {"latitude": 12.9716, "longitude": 77.5946, "accuracy": 30}}
flat = {"latitude": "12.9716", "longitude": "77.5946"}
assert parse_browser_location(nested, now=5.0).accuracy_m == 30.0
assert parse_browser_location(flat, now=5.0).lat == 12.9716
assert parse_browser_location(None) is None
assert parse_browser_location({"error": {"code": 1}}) is None   # permission denied
assert parse_browser_location({"coords": {"latitude": 99, "longitude": 1}}) is None

# 4. IP lookup parsing
good = {"latitude": 12.97, "longitude": 77.59, "city": "Bengaluru",
        "region": "Karnataka", "country_name": "India"}
ip = fetch_ip_location(opener=opener_returning(good), now=7.0)
assert ip.source == SOURCE_IP and ip.place == "Bengaluru, Karnataka, India"
assert fetch_ip_location(opener=opener_returning({"error": True, "reason": "RateLimited"})) is None
assert fetch_ip_location(opener=opener_returning(b"not json")) is None
assert fetch_ip_location(opener=opener_returning([1, 2, 3])) is None
assert fetch_ip_location(opener=failing_opener) is None          # network down
assert fetch_ip_location(opener=opener_returning(good), url="file:///etc/passwd") is None
assert fetch_ip_location(opener=opener_returning(good), url="http://example.com") is None

# 5. Provider priority: fresh browser beats IP
ip_loc = make_location(12.0, 77.0, SOURCE_IP, now=0.0)
fetcher = CountingFetcher([ip_loc])
p = LocationProvider(ip_fetcher=fetcher)
assert p.update_from_browser(nested, now=100.0) is True
assert p.current(now=200.0).source == SOURCE_BROWSER and fetcher.calls == 0
assert p.current_precise(now=200.0).source == SOURCE_BROWSER

# A fresh direct browser payload replaces an IP fallback for current location.
p_direct = LocationProvider(ip_fetcher=CountingFetcher([ip_loc]))
assert p_direct.current(now=100.0).source == SOURCE_IP
direct = {"latitude": 12.9716, "longitude": 77.5946, "accuracy": 18.0}
assert p_direct.update_from_browser(direct, now=101.0) is True
assert p_direct.current(now=102.0).source == SOURCE_BROWSER
assert p_direct.current(now=102.0).accuracy_m == 18.0

# 6. A stale browser location falls back to IP
stale_time = 100.0 + BROWSER_MAX_AGE_SEC + 1
assert p.current(now=stale_time).source == SOURCE_IP and fetcher.calls == 1

# 7. IP results are cached
assert p.current(now=stale_time + 30).source == SOURCE_IP and fetcher.calls == 1

# 8. Manual coordinates override everything, until cleared
assert p.set_manual(10.0, 76.0, now=stale_time) is True
assert p.current(now=stale_time).source == SOURCE_MANUAL
assert p.set_manual(500, 76.0) is False
p.clear_manual()
assert p.current(now=stale_time + 31).source == SOURCE_IP

# 9. A failed IP lookup is not retried at once, and the stale browser is the fallback
fetcher2 = CountingFetcher([])                       # always fails
p2 = LocationProvider(ip_fetcher=fetcher2)
assert p2.current(now=0.0) is None and fetcher2.calls == 1
assert p2.current(now=IP_RETRY_SEC - 1) is None and fetcher2.calls == 1   # throttled
assert p2.current(now=IP_RETRY_SEC + 1) is None and fetcher2.calls == 2   # retried
p2.update_from_browser(flat, now=1000.0)
stale = p2.current(now=1000.0 + BROWSER_MAX_AGE_SEC + 10)
assert stale is not None and stale.source == SOURCE_BROWSER

# 10. An expired IP result is refreshed, and an old one survives a failed refresh
fresh_ip = make_location(13.0, 77.0, SOURCE_IP, now=0.0)
fetcher3 = CountingFetcher([fresh_ip])               # one success, then failures
p3 = LocationProvider(ip_fetcher=fetcher3)
assert p3.current(now=0.0).lat == 13.0
assert p3.current(now=IP_CACHE_SEC + 1).lat == 13.0 and fetcher3.calls == 2

print("All location tests passed")
