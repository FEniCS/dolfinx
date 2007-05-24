"""Unit tests for the mesh library"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2006-08-08 -- 2007-05-23"
__copyright__ = "Copyright (C) 2006 Anders Logg"
__license__  = "GNU GPL Version 2"

import unittest
from dolfin import *

class SimpleShapes(unittest.TestCase):

    def testUnitSquare(self):
        """Create mesh of unit square"""
        mesh = UnitSquare(5, 7)
        self.assertEqual(mesh.numVertices(), 48)
        self.assertEqual(mesh.numCells(), 70)

    def testUnitCube(self):
        """Create mesh of unit cube"""
        mesh = UnitCube(5, 7, 9)
        self.assertEqual(mesh.numVertices(), 480)
        self.assertEqual(mesh.numCells(), 1890)

class MeshRefinement(unittest.TestCase):

    def testRefineUnitSquare(self):
        """Refine mesh of unit square"""
        mesh = UnitSquare(5, 7)
        mesh.refine()
        self.assertEqual(mesh.numVertices(), 165)
        self.assertEqual(mesh.numCells(), 280)

    def testRefineUnitCube(self):
        """Refine mesh of unit cube"""
        mesh = UnitCube(5, 7, 9)
        mesh.refine()
        self.assertEqual(mesh.numVertices(), 3135)
        self.assertEqual(mesh.numCells(), 15120)

class MeshIterators(unittest.TestCase):

    def testVertexIterators(self):
        """Iterate over vertices"""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for v in vertices(mesh):
            n += 1
        self.assertEqual(n, mesh.numVertices())

    def testEdgeIterators(self):
        """Iterate over edges"""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for e in edges(mesh):
            n += 1
        self.assertEqual(n, mesh.numEdges())

    def testFaceIterators(self):
        """Iterate over faces"""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for f in faces(mesh):
            n += 1
        self.assertEqual(n, mesh.numFaces())

    def testFacetIterators(self):
        """Iterate over facets"""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for f in facets(mesh):
            n += 1
        self.assertEqual(n, mesh.numFacets())

    def testCellIterators(self):
        """Iterate over cells"""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for c in cells(mesh):
            n += 1
        self.assertEqual(n, mesh.numCells())
        
    def testMixedIterators(self):
        """Iterate over vertices of cells"""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for c in cells(mesh):
            for v in vertices(c):
                n += 1
        self.assertEqual(n, 4*mesh.numCells())

class BoundaryExtraction(unittest.TestCase):

     def testBoundaryComputation(self):
         """Compute boundary of mesh"""
         mesh = UnitCube(2, 2, 2)
         boundary = BoundaryMesh(mesh)
         self.assertEqual(boundary.numVertices(), 26)
         self.assertEqual(boundary.numCells(), 48)

     def testBoundaryBoundary(self):
         """Compute boundary of boundary"""
         mesh = UnitCube(2, 2, 2)
         b0 = BoundaryMesh(mesh)
         b1 = BoundaryMesh(b0)
         self.assertEqual(b1.numVertices(), 0)
         self.assertEqual(b1.numCells(), 0)

class MeshFunctions(unittest.TestCase):

     def testAssign(self):
         """Assign value of mesh function"""
         mesh = UnitSquare(3, 3)
         f = MeshFunction('int', mesh, 0)
         f.set(3, 10)
         v = Vertex(mesh, 3)
         self.assertEqual(f(v), 10)
      
class InputOutput(unittest.TestCase):

     def testMeshXML2D(self):
         """Write and read 2D mesh to/from file"""
         mesh_out = UnitSquare(3, 3)
         mesh_in  = Mesh()
         file = File("unitsquare.xml")
         file << mesh_out
         file >> mesh_in
         self.assertEqual(mesh_in.numVertices(), 16)

     def testMeshXML3D(self):
         """Write and read 3D mesh to/from file"""
         mesh_out = UnitCube(3, 3, 3)
         mesh_in  = Mesh()
         file = File("unitcube.xml")
         file << mesh_out
         file >> mesh_in
         self.assertEqual(mesh_in.numVertices(), 64)

     def testMeshMatlab2D(self):
         """Write matlab format (no real test)"""
         mesh = UnitSquare(5, 5)
         file = File("unitsquare.m")
         file << mesh
         self.assertEqual(0, 0)

     def testMeshFunction(self):
         """Write and read mesh function to/from file"""
         mesh = UnitSquare(1, 1)
         f = MeshFunction('int', mesh, 0)
         f.set(0, 2)
         f.set(1, 4)
         f.set(2, 6)
         f.set(3, 8)
         file = File("meshfunction.xml")
         file << f
         g = MeshFunction('int', mesh, 0)
         file >> g
         for v in vertices(mesh):
             self.assertEqual(f(v), g(v))

class PyCCInterface(unittest.TestCase):

     def testGetGeometricalDimension(self):
         """Get geometrical dimension of mesh"""
         mesh = UnitSquare(5, 5)
         self.assertEqual(mesh.geometry().dim(), 2)

     def testGetCoordinates(self):
         """Get coordinates of vertices"""
         mesh = UnitSquare(5, 5)
         self.assertEqual(len(mesh.coordinates()), 36)

     def testGetCells(self):
         """Get cells of mesh"""
         mesh = UnitSquare(5, 5)
         self.assertEqual(len(mesh.cells()), 50)

if __name__ == "__main__":
    unittest.main()
