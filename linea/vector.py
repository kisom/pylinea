import math
import numpy
import pdb

EQUALITY_TOLERANCE = 0.001


class NonConformantVectors(Exception):
    def __init__(self, expected, actual):
        self.expected = expected
        self.actual = actual

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

    def unit(self):
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("cannot normalise the zero vector")
        return self * (1 / mag)

# The dot (or inner) product determines the angle between two vectors.
def dot(v, w):
    assert(isinstance(v, Vector))
    assert(isinstance(v, Vector))
    inner = sum([a * b for (a, b) in zip(v.v, w.v)])

    # Cauchy-Schwartz inequality
    assert(abs(inner) <= (v.magnitude() * w.magnitude()))
    return inner

def angle(v, w):
    assert(isinstance(v, Vector))
    assert(isinstance(v, Vector))

    inner = dot(v.unit(), w.unit())
    theta = math.acos(inner)
    return theta

def r2d(rval):
    return rval * 180 / math.pi