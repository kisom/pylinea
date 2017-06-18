import linea.vector as vec
import linea.util as util
import numpy
import pytest
from .context import vec


def fequal(a, b):
    eq = numpy.isclose(a, b, util.EQUALITY_TOLERANCE)
    if isinstance(eq, numpy.ndarray):
        return eq.all()
    return eq


def test_equality():
    v1 = vec.Vector(1, 2, 3)
    v2 = vec.Vector(1, 2)
    v3 = vec.Vector(1, 2, 3)
    v4 = vec.Vector(3, 4, 5)

    assert (v1 == v3)
    with pytest.raises(vec.NonConformantVectors):
        assert v1 != v2
    assert (v1 != v4)


# Video 4
def test_basic_operations():
    v1 = vec.Vector(8.218, -9.341)
    v2 = vec.Vector(-1.129, 2.111)
    v3 = vec.Vector(7.089, -7.230)
    assert (v1 + v2 == v3)

    v1 = vec.Vector(7.119, 8.215)
    v2 = vec.Vector(-8.223, 0.878)
    v3 = vec.Vector(15.3420, 7.3370)
    assert (v1 - v2 == v3)

    v1 = vec.Vector(1.671, -1.012, -0.318)
    k = 7.41
    v2 = vec.Vector(12.3821, -7.4989, -2.3564)
    assert (k * v1 == v2)
    assert (v1 * k == v2)


# Video 6
def test_magnitude():
    v1 = vec.Vector(-0.221, 7.437)
    assert (fequal(v1.magnitude(), 7.4403))

    v2 = vec.Vector(5.581, -2.136)
    unit = vec.Vector(0.933935214087, -0.357442325262)
    assert (v2.unit() == unit)

    v3 = vec.Vector(8.813, -1.331, -6.247)
    assert (fequal(v3.magnitude(), 10.8842))

    v4 = vec.Vector(1.996, 3.108, -4.554)
    unit = vec.Vector(.340401295943, 0.530043701298, -0.776647044953)
    assert (v4.unit() == unit)

    v5 = vec.Vector(0, 0, 0)
    with pytest.raises(ValueError):
        v5.unit()


# Video 8
def test_dot_product():
    v1 = vec.Vector(1, 2, -1)
    v2 = vec.Vector(3, 1, 0)
    assert (fequal(vec.dot(v1, v2), 5))
    assert (fequal(v1.dot(v2), 5))

    v1 = vec.Vector(7.887, 4.138)
    v2 = vec.Vector(-8.802, 6.776)
    assert (fequal(vec.dot(v1, v2), -41.382))

    v3 = vec.Vector(-5.955, -4.904, -1.874)
    v4 = vec.Vector(-4.496, -8.755, 7.103)
    assert (fequal(vec.dot(v3, v4), 56.3971))

    v5 = vec.Vector(3.183, -7.627)
    v6 = vec.Vector(-2.668, 5.319)
    assert (fequal(vec.angle(v5, v6), 3.072))

    v7 = vec.Vector(7.35, 0.221, 5.188)
    v8 = vec.Vector(2.751, 8.259, 3.985)
    theta = vec.angle(v7, v8, in_degrees=True)
    assert (fequal(theta, 60.2758))

    v9 = vec.Vector(0, 0)
    with pytest.raises(ValueError):
        v1.angle_with(v9)


# Video 10.
def test_parallel_orthogonal():
    v1 = vec.Vector(-7.579, -7.88)
    v2 = vec.Vector(22.737, 23.64)
    assert (v1.parallel_to(v2))
    assert (not v1.orthogonal_to(v2))

    v3 = vec.Vector(-2.029, 9.97, 4.172)
    v4 = vec.Vector(-9.231, -6.639, -7.245)
    assert (not v3.parallel_to(v4))
    assert (not v3.orthogonal_to(v4))

    v5 = vec.Vector(-2.328, -7.284, -1.214)
    v6 = vec.Vector(-1.821, 1.072, -2.94)
    assert (not v5.parallel_to(v6))
    assert (v5.orthogonal_to(v6))

    v7 = vec.Vector(2.118, 4.827)
    v8 = vec.Vector(0, 0)
    assert v7.parallel_to(v8)
    assert v7.orthogonal_to(v8)


# Video 12
def test_projection():
    # sanity check
    v1 = vec.Vector(1, 3)
    v2 = vec.Vector(3, 3)
    v3 = v1.project_parallel(v2)
    v4 = v1.project_orthogonal(v2)
    assert v3 + v4 == v1

    v1 = vec.Vector(3.039, 1.879)
    v2 = vec.Vector(0.825, 2.036)
    v3 = vec.Vector(1.0826, 2.6717)
    assert v1.project_parallel(v2) == v3

    v4 = vec.Vector(-9.88, -3.264, -8.159)
    v5 = vec.Vector(-2.155, -9.353, -9.473)
    v6 = vec.Vector(-8.350, 3.376, -1.434)
    assert v4.project_orthogonal(v5) == v6

    v7 = vec.Vector(3.009, -6.172, 3.692, -2.510)
    v8 = vec.Vector(6.404, -9.144, 2.759, 8.718)
    v9 = vec.Vector(1.969, -2.811, 0.848, 2.680)
    assert v7.project_parallel(v8) == v9
    v10 = vec.Vector(1.040, -3.361, 2.844, -5.190)
    assert v7.project_orthogonal(v8) == v10

    assert (v9 + v10) == v7

def test_cross_product():
    # sanity check
    v1 = vec.Vector(5, 3, -2)
    v2 = vec.Vector(-1, 0, 3)
    v3 = vec.Vector(9, -13, 3)
    assert vec.cross(v1, v2) == v3

    v1 = vec.Vector(8.462, 7.893, -8.187)
    v2 = vec.Vector(6.984, -5.975, 4.778)
    v3 = vec.Vector(-11.205, -97.609, -105.685)
    assert vec.cross(v1, v2) == v3

    v4 = vec.Vector(-8.987, -9.838, 5.031)
    v5 = vec.Vector(-4.268, -1.861, -8.866)
    area = 142.122
    assert fequal(vec.area_parallelogram(v4, v5), area)

    v6 = vec.Vector(1.500, 9.547, 3.691)
    v7 = vec.Vector(-6.007, 0.124, 5.772)
    area = 42.565
    assert fequal(vec.area_triangle(v6, v7), area)