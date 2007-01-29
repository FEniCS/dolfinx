"""Unit test for the mesh library"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2006-08-08 -- 2006-11-28"
__copyright__ = "Copyright (C) 2006 Anders Logg"
__license__  = "GNU GPL Version 2"

import unittest
from dolfin import *

class BasisFunctions(unittest.TestCase):

    def testBasisFunctions(self):
        
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, "triangle", 2, 2)
        editor.initVertices(3)
        editor.addVertex(0, 0.4, 0.4)
        editor.addVertex(1, 0.7, 0.4)
        editor.addVertex(2, 0.4, 0.7)
        editor.initCells(1)
        editor.addCell(0, 0, 1, 2)
        editor.close()

        cell = Cell(mesh, 0)

        v = vertices(cell)
        p0 = v.point()
        v.increment()
        p1 = v.point()
        v.increment()
        p2 = v.point()

        pmid = p0 + p1 + p2
        pmid = pmid * (1.0 / 3.0)

        print ""
        print "Cell:"
        print "p0: ", p0[0], " ", p0[1], " ", p0[2]
        print "p1: ", p1[0], " ", p1[1], " ", p1[2]
        print "p2: ", p2[0], " ", p2[1], " ", p2[2]
        print "pmid: ", pmid[0], " ", pmid[1], " ", pmid[2]

        element = P1tri()
        basis = FEBasis()
        basis.construct(element)
        map = NewAffineMap()
        map.update(cell)

        print "Evaluating basis functions at vertices and midpoint: "

        phi = [];

        for i in range(0, basis.functions.size()):
            val_0 = basis.evalPhysical(basis.functions[i], p0, map, 0)
            val_1 = basis.evalPhysical(basis.functions[i], p1, map, 0)
            val_2 = basis.evalPhysical(basis.functions[i], p2, map, 0)
            val_mid = basis.evalPhysical(basis.functions[i], pmid, map, 0)

            phi.append([val_0, val_1, val_2, val_mid])


        for i in range(0, basis.functions.size()):

            print "phi_%i" % i, ": ", phi[i]

if __name__ == "__main__":
    unittest.main()
