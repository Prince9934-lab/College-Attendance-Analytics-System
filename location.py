# location.py

import math
from typing import Tuple, Optional
import requests

from config import config


def haversine_distance(
    coord1: Tuple[float, float],
    coord2: Tuple[float, float]
) -> float:
    """
    Calculate distance between two coordinates in meters
    """

    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))

    r = 6371000  # Earth radius in meters

    return c * r


# -----------------------------
# TEMPORARY TEST LOCATION
# -----------------------------
def get_gps_location():
    """
    Temporary fixed location for testing
    """
    return (23.2599, 77.4126)


# -----------------------------
# REAL IP LOCATION
# -----------------------------
def get_ip_location():
    """
    Get current location using IP
    """

    try:
        response = requests.get(
            "https://ipinfo.io/json",
            timeout=5
        )

        if response.status_code == 200:

            data = response.json()

            loc = data.get("loc")

            if loc:
                lat, lon = map(float, loc.split(","))

                return (lat, lon)

    except Exception as e:
        print("Location Error:", e)

    return None


# -----------------------------
# LOCATION VALIDATION
# -----------------------------
def validate_location(
    current_location: Optional[Tuple[float, float]] = None
) -> Tuple[bool, str, float]:

    if current_location is None:
        current_location = get_gps_location()

    if current_location is None:
        return False, "Could not determine location", -1

    distance = haversine_distance(
        current_location,
        config.CAMPUS_LOCATION
    )

    if distance <= config.ALLOWED_RADIUS_METERS:

        return (
            True,
            f"Location valid ({distance:.1f}m from campus)",
            distance
        )

    return (
        False,
        f"Too far from campus ({distance:.1f}m)",
        distance
    )


# -----------------------------
# LOCATION VALIDATOR CLASS
# -----------------------------
class LocationValidator:

    def __init__(self):

        self._cached_location = None
        self._cache_valid = False

    def refresh_location(self):

        self._cached_location = get_gps_location()

        self._cache_valid = (
            self._cached_location is not None
        )

    def validate(self):

        if not self._cache_valid:
            self.refresh_location()

        return validate_location(
            self._cached_location
        )