import math
import numpy

EQUALITY_TOLERANCE = 0.001


class NonConformantVectors(Exception):
    def __init__(self, expected, actual):
        self.expected = expected
        self.actual = actual
        Exception.__init__(self)

    def __str__(self):
        msg = "Expected the right-hand vector to have dimension {}, "
        msg += "but it has a dimension of {}."
        return msg.format(self.expected, self.actual)


class Vector:
    def __init__(self, a=None, *args):
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
        return numpy.isclose(self.v, other.v, EQUALITY_TOLERANCE).all()

    def __repr__(self):
        return 'Vector[{}]'.format(len(self))

    def magnitude(self):
        return math.sqrt(sum(map(lambda x: x * x, self.v)))

    def is_zero(self):
        return numpy.isclose(self.magnitude(), 0, EQUALITY_TOLERANCE)

    def unit(self):
        mag = self.magnitude()
        if self.is_zero():
            raise ValueError("cannot normalise the zero vector")
        return self * (1 / mag)

    def dot(self, other):
        return dot(self, other)

    def angle_with(self, other, in_degrees=False):
        return angle(self, other, in_degrees=in_degrees)

    def parallel_to(self, other):
        return parallel(self, other)

    def orthogonal_to(self, other):
        return orthogonal(self, other)


# The dot (or inner) product determines the angle between two vectors.
def dot(v, w):
    assert isinstance(v, Vector)
    assert isinstance(v, Vector)
    inner = sum([a * b for (a, b) in zip(v.v, w.v)])

    # Cauchy-Schwartz inequality
    assert abs(inner) <= (v.magnitude() * w.magnitude())
    return inner


def angle(v, w, in_degrees=False):
    assert isinstance(v, Vector)
    assert isinstance(v, Vector)

    try:
        # check for floating point problems resulting in domain errors
        inner = dot(v.unit(), w.unit())
        if inner > 1:
            if numpy.isclose(inner, 1, 0.00000001):
                inner = 1
        if inner < -1:
            if numpy.isclose(inner, -1, 0.00000001):
                inner = -1
    except NonConformantVectors:
        raise ValueError('Cannot determine the angle between the zero vector and another vector.')
    except:
        # Deliberately pass through any other exceptions.
        raise
    theta = math.acos(inner)
    if in_degrees:
        theta = r2d(theta)
    return theta


def parallel(v, w):
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
    if numpy.isclose(theta, 0, EQUALITY_TOLERANCE):
        return True
    return numpy.isclose(theta, math.pi, EQUALITY_TOLERANCE)


def orthogonal(v, w):
    if len(v) != len(w):
        raise NonConformantVectors(len(v), len(w))
    if v.is_zero() or w.is_zero():
        return True
    return numpy.isclose(dot(v, w), 0, EQUALITY_TOLERANCE)


def r2d(rval):
    assert abs(rval) <= (2 * math.pi)
    return rval * 180 / math.pi
