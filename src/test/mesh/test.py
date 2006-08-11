"""Run all mesh unit tests."""

__author__ = "Garth N. Wells (g.n.wells@tudelft.nl)"
__date__ = "2006-08-11"
__copyright__ = "Copyright (C) 2006 Garth N. Wells"
__license__  = "GNU GPL Version 2"

import unittest

#from mesh import *

def meshTestSuite():
    """Collection of all mesh tests"""    
    tests = unittest.TestSuite()
    tests.addTest(unittest.findTestCases(__import__('unitMesh')))
    print tests
    print " "
    tests.addTest(unittest.findTestCases(__import__('createUBlas')))
    print tests

    return tests

if __name__ == '__main__':
    unittest.main(defaultTest='meshTestSuite')

