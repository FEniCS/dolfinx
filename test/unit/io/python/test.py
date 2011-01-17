"""Unit tests for the io library"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2009-01-02 -- 2009-01-02"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

import unittest
from dolfin import *

class VectorXML_IO(unittest.TestCase):

    def testSaveReadVector(self):
        size = 512
        x = Vector(size)
        x[:] = 1.0

        out_file = File("test_vector_xml.xml")
        out_file << x

        y = Vector()
        out_file >> y
        self.assertEqual(x.size(), y.size())
        self.assertAlmostEqual((x - y).norm("l2"), 0.0)


if __name__ == "__main__":
    unittest.main()
