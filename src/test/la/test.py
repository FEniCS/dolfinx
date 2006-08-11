"""Run all la unit tests."""

__author__ = "Garth N. Wells (g.n.wells@tudelft.nl)"
__date__ = "2006-08-11"
__copyright__ = "Copyright (C) 2006 Garth N. Wells"
__license__  = "GNU GPL Version 2"

import unittest

from createUBlas import *
from algebraUBlas import *
from solverUBlas import *

def laTestSuite():
    """Collection of all la tests"""    
    tests = unittest.TestSuite()
    tests.addTest(unittest.findTestCases(__import__('createUBlas')))
    tests.addTest(unittest.findTestCases(__import__('algebraUBlas')))
    tests.addTest(unittest.findTestCases(__import__('solverUBlas')))

    return tests

if __name__ == '__main__':
    unittest.main(defaultTest='laTestSuite')

