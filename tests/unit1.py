import linea.vector as vec
import numpy
import unittest
from .context import vec

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_equality(self):
        v1 = vec.Vector(1, 2, 3)
        v2 = vec.Vector(1, 2)
        v3 = vec.Vector(1, 2, 3)
        v4 = vec.Vector(3, 4, 5)

        self.assertEqual(v1, v3)
        with self.assertRaises(vec.NonConformantVectors):
            v1 == v2
        self.assertFalse(v1 == v4)

    # Video 4
    def test_basic_operations(self):
        v1 = vec.Vector(8.218, -9.341)
        v2 = vec.Vector(-1.129, 2.111)
        v3 = vec.Vector(7.089, -7.230)
        self.assertEqual(v1 + v2, v3)

        v1 = vec.Vector(7.119, 8.215)
        v2 = vec.Vector(-8.223, 0.878)
        v3 = vec.Vector(15.3420, 7.3370)
        self.assertEqual(v1 - v2, v3)

        v1 = vec.Vector(1.671, -1.012, -0.318)
        k = 7.41
        v2 = vec.Vector(12.3821, -7.4989, -2.3564)
        self.assertEqual(k * v1, v2)

    # Video 6
    def test_magnitude(self):
        v1 = vec.Vector(-0.221, 7.437)
        self.assertTrue(numpy.isclose(v1.magnitude(), 7.4403, vec.EQUALITY_TOLERANCE))

        v2 = vec.Vector(5.581, -2.136)
        unit = vec.Vector(0.933935214087, -0.357442325262)
        self.assertEqual(v2.unit(), unit)

        v3 = vec.Vector(8.813, -1.331, -6.247)
        self.assertTrue(numpy.isclose(v3.magnitude(), 10.8842, vec.EQUALITY_TOLERANCE))

        v4 = vec.Vector(1.996, 3.108, -4.554)
        unit = vec.Vector(.340401295943, 0.530043701298, -0.776647044953)
        self.assertEqual(v4.unit(), unit)

        v5 = vec.Vector(0, 0, 0)
        with self.assertRaises(ValueError):
            v5.unit()

if __name__ == '__main__':
    unittest.main()