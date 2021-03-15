import unittest
from unittest import TestCase

import numpy as np

from my_network import initialize_parameters, linear_forward, relu


class Test(unittest.TestCase):
    def test_initialize_parameters(self):
        nets = initialize_parameters([4, 6, 6, 10])
        a = np.random.uniform(0, 255, 4).reshape(-1, 1)  # mnist image

        z, linear_cache = linear_forward(a, nets.get('W1'), nets.get('b1'))
        self.assertTrue(z.size, 6)

        a = z
        z, linear_cache = linear_forward(a, nets.get('W2'), nets.get('b2'))
        self.assertTrue(z.size, 6)

        a = z
        z, linear_cache = linear_forward(a, nets.get('W3'), nets.get('b3'))
        self.assertTrue(z.size, 10)

    def test_relu(self):
        Z = np.array([[-1], [2]])
        A, _ = relu(Z)

        expected_relu = np.array([[0], [2]])
        self.assertTrue((A == expected_relu).all())

        self.assertTrue(A.size == Z.size)


if __name__ == '__main__':
    unittest.main()
