"""Unit tests for the linear algebra library"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2006-08-09 -- 2006-08-09"
__copyright__ = "Copyright (C) 2006 Anders Logg"
__license__  = "GNU GPL Version 2"

import unittest
from dolfin import *

class CreateVectors(unittest.TestCase):

    def testUBlasVector(self):
        """Create uBlas vector"""
        x = uBlasVector(10)
        self.assertEqual(x.size(), 10)

if __name__ == "__main__":
    unittest.main()
