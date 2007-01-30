"""Unit test for the mesh library"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2006-08-08 -- 2006-11-28"
__copyright__ = "Copyright (C) 2006 Anders Logg"
__license__  = "GNU GPL Version 2"

import unittest
from dolfin import *

class Functions(unittest.TestCase):

    def testEvaluation(self):
        
        mesh = UnitSquare(2, 2)
        element = P1tri()
        N = FEM_size(mesh, element)
        x = Vector(N)

        U = Function(x, mesh, element)

        for i in range(0, N):
            x[i] = i
        
        for v in vertices(mesh):
            id = v.index()
            p = v.point()
            x[id] = p[0] + p[1]


        probex = 0.66
        probey = 0.833

        pprobe = Point(probex, probey)

        val = U(pprobe)

        print "val: ", val
        ref = probex + probey
        print "ref: ", ref

        self.assertEqual(val, ref)

if __name__ == "__main__":
    unittest.main()
