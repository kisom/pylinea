# -*- coding: utf-8 -*-
"""
``linea.util``

Utility math functions.
"""
import math
import numpy

EQUALITY_TOLERANCE = 0.001


def r2d(rval):
    """
    Convert the integer or floating point radian value to degrees. The
    radian value must be between :math:`-2*\pi <= rval <= 2*\pi`.

    >>> r2d(math.pi)
    180.0
    >>> r2d(3 * math.pi / 4)
    135.0
    >>> r2d(3 * math.pi)
    ---------------------------------------------------------------------------
    AssertionError                            Traceback (most recent call last)
    """
    assert abs(rval) <= (2 * math.pi)
    return rval * 180 / math.pi


def clamp_if_close(value, clamped, tolerance=EQUALITY_TOLERANCE):
    """
    Return clamped if value is close to clamped (within tolerance), or return
    value otherwise.

    >>> clamp_if_close(0.99, 1.0, tolerance=0.1)
    1.0
    >>> clamp_if_close(0.99, 1.0, tolerance=0.001)
    0.99
    """
    if numpy.isclose(value, clamped, tolerance):
        return clamped
    return value
