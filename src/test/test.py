"""Run all unit tests."""

__author__ = "Garth N. Wells (g.n.wells@tudelft.nl)"
__date__ = "2006-08-11"
__copyright__ = "Copyright (C) 2006 Garth N. Wells"
__license__  = "GNU GPL Version 2"

import unittest

import sys
sys.path.append('./la')
sys.path.append('./mesh')

from la.test import laTestSuite
from mesh.test import meshTestSuite

def TestSuite():
    """Collection of all test suites"""
    tests = unittest.TestSuite()
    # Add test suites here
    tests.addTest(laTestSuite())
    tests.addTest(meshTestSuite())
    return tests   

if __name__ == '__main__':
    unittest.main(defaultTest='TestSuite')

