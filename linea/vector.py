"""
pylinea.vector implements arbitrarily-sized single-dimension vectors
built on numpy's ndarray.
"""
# pylint: disable=C0103
import math
import numpy

from . import util


class NonConformantVectors(Exception):
    """
    A NonConformantVector is thrown when attempting to do operations
    with vectors of differing sizes; generally, the size of the
    /right-hand/ argument doesn't conform to the size of the /left-hand/
    argument.
    """

    def __init__(self, expected, actual):
        self.expected = expected
        self.actual = actual
        Exception.__init__(self)

    def __str__(self):
        msg = "Expected the right-hand vector to have dimension {}, "
        msg += "but it has a dimension of {}."
        return msg.format(self.expected, self.actual)


class Vector:
    """
    A vector is a one-dimensional vector of some arbitrary size. This size
    is fixed and can't be changed later in the Vector's life.
    """

    def __init__(self, a=None, *args):
        """
        Initialise a vector, either using an iterable passed in or as a sequence
        of values.
        >>> print(Vector(1, 2, 3))
        [1; 2; 3]
        >>> print(Vector([1, 2, 3]))
        [1; 2; 3]
        """
        if len(args) > 0:
            a = [a]
            a.extend(args)
            self.v = numpy.array(a)
        else:
            if isinstance(a, numpy.ndarray):
                self.v = a
            else:
                self.v = numpy.array(a)

    def __str__(self):
        s = '['
        for i in range(len(self.v)):
            s += str(self.v[i])
            if i < len(self.v) - 1:
                s += '; '
        s += ']'
        return s

    def __len__(self):
        return len(self.v)

    def __iter__(self):
        return (x for x in self.v)

    def __mul__(self, other):
        o = list(map(lambda x: x * other, self.v))
        return Vector(a=o)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if len(self) != len(other):
            raise NonConformantVectors(len(self), len(other))
        return Vector(self.v + other.v)

    def __sub__(self, other):
        if len(self) != len(other):
            raise NonConformantVectors(len(self), len(other))
        return Vector(self.v - other.v)

    def __eq__(self, other):
        if not isinstance(other, Vector):
            raise ValueError
        if len(self) != len(other):
            raise NonConformantVectors(len(self), len(other))
        eq = numpy.isclose(self.v, other.v, util.EQUALITY_TOLERANCE)
        if isinstance(eq, bool):
            return eq
        return eq.all()

    def __repr__(self):
        return 'Vector[{}]'.format(len(self))

    def magnitude(self):
        """
        Return the magnitude of the vector.
        """
        return math.sqrt(sum(map(lambda x: x * x, self.v)))

    def is_zero(self, tolerance=util.EQUALITY_TOLERANCE):
        """
        Return True if the vector is a zero vector (within some tolerance).
        """
        return numpy.isclose(self.magnitude(), 0, tolerance)

    def unit(self):
        """
        Return the unit vector of this vector. If this method is called on
        a zero vector (i.e. is_zero returns True), a ValueError will be thrown.
        """
        mag = self.magnitude()
        if self.is_zero():
            raise ValueError("cannot normalise the zero vector")
        return self * (1 / mag)

    def dot(self, other):
        """
        Compute the dot product between this vector and the other vector.
        """
        return dot(self, other)

    def angle_with(self, other, in_degrees=False):
        """
        Compute the angle between this vector and the other vector. The
        default is to return the angle in radians. If in_degrees is True,
        returns the angle in degrees.
        """
        return angle(self, other, in_degrees=in_degrees)

    def parallel_to(self, other):
        """Return True if the vector other is parallel to this vector."""
        return parallel(self, other)

    def orthogonal_to(self, other):
        """"Return True if the vector other is orthogonal to this vector. """
        return orthogonal(self, other)


# The dot (or inner) product determines the angle between two vectors.
def dot(v, w):
    """
    Return the dot product of vectors v and w.
    """
    assert isinstance(v, Vector)
    assert isinstance(v, Vector)
    inner = sum([a * b for (a, b) in zip(v.v, w.v)])

    # Cauchy-Schwartz inequality
    assert abs(inner) <= (v.magnitude() * w.magnitude())
    return inner


def angle(v, w, in_degrees=False, tolerance=util.EQUALITY_TOLERANCE):
    """
    Return the angle between vectors v and w in radians. If in_degrees is
    True, return the answer in degrees.
    """
    assert isinstance(v, Vector)
    assert isinstance(v, Vector)

    try:
        # check for floating point problems resulting in domain errors
        inner = dot(v.unit(), w.unit())
        if inner > 1:
            inner = util.clamp_if_close(inner, 1.0, tolerance)
        if inner < -1:
            inner = util.clamp_if_close(inner, -1.0, tolerance)
    except NonConformantVectors:
        raise ValueError('Cannot determine the angle between the zero vector and another vector.')
    except:
        # Deliberately pass through any other exceptions.
        raise
    theta = math.acos(inner)
    if in_degrees:
        theta = util.r2d(theta)
    return theta


def parallel(v, w, tolerance=util.EQUALITY_TOLERANCE):
    """Return True if vectors v and w are parallel."""
    if len(v) != len(w):
        raise NonConformantVectors(len(v), len(w))
    if v.is_zero() or w.is_zero():
        return True

    # HERE LIES THE RUIN OF A DUMB
    # I'd originally tried to code this by creating a list of v_i / w_i, and then reducing
    # the list via comparison (e.g. are all the values in the list numpy.isclose?). I got it
    # right, but... the video showed a better way. Lesson learned, think about things instead
    # of blindly coding through.
    theta = angle(v, w)
    if numpy.isclose(theta, 0, tolerance):
        return True
    return numpy.isclose(theta, math.pi, tolerance)


def orthogonal(v, w, tolerance=util.EQUALITY_TOLERANCE):
    """Return True if vectors v and w are orthogonal."""
    if len(v) != len(w):
        raise NonConformantVectors(len(v), len(w))
    if v.is_zero() or w.is_zero():
        return True
    return numpy.isclose(dot(v, w), 0, tolerance)
