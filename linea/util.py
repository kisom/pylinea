"""
Utility math functions.
"""
import math
import numpy

EQUALITY_TOLERANCE = 0.001


def r2d(rval):
    """Convert radians to degrees."""
    assert abs(rval) <= (2 * math.pi)
    return rval * 180 / math.pi


def clamp_if_close(value, clamped, tolerance=EQUALITY_TOLERANCE):
    """
    Return clamped if value is close to clamped (within tolerance), or return
    value otherwise.
    """
    if numpy.isclose(value, clamped, tolerance):
        return clamped
    return value
