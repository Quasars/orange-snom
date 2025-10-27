import os
import unittest


def load_tests(loader, tests, pattern):
    tests_dir = os.path.dirname(__file__)

    if loader is None:
        loader = unittest.TestLoader()
    if pattern is None:
        pattern = 'test*.py'

    all_tests = [
        loader.discover(tests_dir, pattern, tests_dir),
    ]

    return unittest.TestSuite(all_tests)
