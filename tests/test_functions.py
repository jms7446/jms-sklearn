import numpy as np
from unittest import TestCase, main

import functions


class FunctionsTest(TestCase):

    def test_get_one_hot(self):
        np.testing.assert_equal(functions.get_one_hot(2, 5), np.array([0, 0, 1, 0, 0]))
        np.testing.assert_equal(functions.get_one_hots([0, 2], 5), np.array([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]]))


if __name__ == "__main__":
    main()
