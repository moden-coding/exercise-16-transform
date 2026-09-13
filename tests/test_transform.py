#!/usr/bin/env python3
"""Tests for the Transform assignment."""

import unittest
from unittest.mock import patch

import numpy as np

from src.transform import transform


class TestTransform(unittest.TestCase):
    """transform(s1, s2) -> list of elementwise products of two number strings."""

    def test_worked_example(self):
        s1 = "1 5 3"
        s2 = "2 6 -1"
        result = transform(s1, s2)
        self.assertIsInstance(
            result,
            list,
            msg="transform should return a list. Got %s." % (type(result),),
        )
        self.assertEqual(
            result,
            [2, 30, -3],
            msg="transform(%r, %r) should be [2, 30, -3]: "
            "1*2=2, 5*6=30, 3*-1=-3." % (s1, s2),
        )

    def test_empty_strings_give_an_empty_list(self):
        result = transform("", "")
        self.assertIsInstance(
            result,
            list,
            msg="transform should return a list. Got %s." % (type(result),),
        )
        self.assertEqual(
            result,
            [],
            msg="transform('', '') should return an empty list. Note that "
            "''.split() and ''.split(' ') behave differently: "
            "''.split(' ') gives [''] while ''.split() gives [].",
        )

    def test_random_values_multiply_elementwise(self):
        L1 = np.random.randint(-100, 100, 50)
        L2 = np.random.randint(-100, 100, 50)
        s1 = " ".join(map(str, L1))
        s2 = " ".join(map(str, L2))
        result = transform(s1, s2)
        self.assertIsInstance(
            result,
            list,
            msg="transform should return a list. Got %s." % (type(result),),
        )
        for a, b, c in zip(L1, L2, result):
            self.assertEqual(
                a * b,
                c,
                msg="Expected %d * %d = %d in the result, got %d."
                % (a, b, a * b, c),
            )

    def test_uses_zip_and_map(self):
        s1 = "1 5 3"
        s2 = "2 6 -1"
        with patch('builtins.zip') as z:
            with patch('builtins.map') as m:
                transform(s1, s2)
                z.assert_called()
                self.assertGreaterEqual(
                    len(m.mock_calls),
                    2,
                    msg="transform should use map() at least twice (e.g. to "
                    "convert each split string of digits to ints).",
                )


if __name__ == '__main__':
    unittest.main()
