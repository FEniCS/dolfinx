"""Unit test for the mesh library"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2006-08-08 -- 2006-11-28"
__copyright__ = "Copyright (C) 2006 Anders Logg"
__license__  = "GNU GPL Version 2"

import unittest
from dolfin import *

class Functions(unittest.TestCase):

    def testVerifyNonMatchingProjection(self):

        meshA = Mesh()
        editor = MeshEditor()
        editor.open(meshA, "triangle", 2, 2)
        editor.initVertices(4)
        editor.addVertex(0, 0.0, 0.0)
        editor.addVertex(1, 1.0, 0.0)
        editor.addVertex(2, 1.0, 1.0)
        editor.addVertex(3, 0.0, 1.0)
        editor.initCells(2)
        editor.addCell(0, 0, 1, 3)
        editor.addCell(1, 1, 2, 3)
        editor.close()

        NA = meshA.numCells()
        xA = Vector(NA)
        
        for v in cells(meshA):
            id = v.index()
            xA[id] = id

        K = DP0tri()
            
        fA = Function(xA, meshA, K)
        
        meshB = UnitSquare(1, 1)
        fN = Function(fA, meshB)
        fN.attach(meshB)

        fileA = File("fA.pvd")
        fileA << fA

        NB = meshB.numCells()
        xB = Vector(NB)

        fB = Function(xB, meshB, K)

        forms = import_formfile("Projection.form")

        a = forms.ProjectionBilinearForm()
        L = forms.ProjectionLinearForm(fN)

        print "element:"
        print a.trial().spec().repr()

        A = Matrix()
        b = Vector()

        FEM_assemble(a, A, meshB)
        FEM_assemble(L, b, meshB)

        Alump = Vector(b.size())

        print "A:"
        A.disp()
        print "b:"
        b.disp()


        #solver = KrylovSolver()
        #solver.solve(A, xB, b)
        xB.copy(b, 0, 0, b.size())
        A.lump(Alump)
        xB.div(Alump)

        print "xB:"
        xB.disp()


        fileB = File("fB.pvd")
        fileB << fB

        Verified = False
        if(abs(xB[0] - 0.5) < 1.0e-8):
            Verified = True

        self.assertEqual(Verified, True)

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
