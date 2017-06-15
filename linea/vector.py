import math
import numpy


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

    def __add__(self, other):
        if len(self) != len(other):
            raise NonConformantVectors(len(self), len(other))
        return Vector(self.v + other.v)

    def __sub__(self, other):
        if len(self) != len(other):
            raise NonConformantVectors(len(self), len(other))
        return Vector(self.v - other.v)

    def __eq__(self, other):
        return (self.v == other.v).all()

    def __repr__(self):
        return 'Vector[{}]'.format(len(self))

    def magnitude(self):
        return math.sqrt(sum(map(lambda x: x * x, self.v)))

    def unit(self):
        return self * (1 / self.magnitude())


def unit1_test():
    # Test Vector string representations.
    v1 = Vector(1, 2, 3)
    print("Vector: {}".format(str(v1)))

    # Test Vector representation.
    v2 = Vector(1, 2, 3)
    print("Vector object is {}".format(repr(v2)))

    # Test equality checks.
    v3 = Vector(1, 3, 5)
    print("{} == {}: {}".format(v1, v2, v1 == v2))
    print("{} == {}: {}".format(v1, v3, v1 == v3))

    v1 = Vector(8.218, -9.341)
    v2 = Vector(-1.129, 2.111)
    print("{} + {} = {}".format(v1, v2, v1 + v2))

    v1 = Vector(7.119, 8.215)
    v2 = Vector(-8.223, 0.878)
    print("{} - {} = {}".format(v1, v2, v1 - v2))

    v1 = Vector(1.671, -1.012, -0.318)
    k = 7.41
    print("{} * {} = {}".format(v1, k, v1 * k))

    v1 = Vector(1, 2, 3)
    print('Magnitude of {}: {}'.format(v1, v1.magnitude()))
    print('Unit vector for {} is {}'.format(v1, v1.unit()))

if __name__ == '__main__':
    unit1_test()
