"""Unit tests for the mesh library"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2006-08-08 -- 2010-11-24"
__copyright__ = "Copyright (C) 2006 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

import unittest
import numpy.random
from dolfin import *

class SimpleShapes(unittest.TestCase):

    def testUnitSquare(self):
        """Create mesh of unit square."""
        mesh = UnitSquare(5, 7)
        self.assertEqual(mesh.num_vertices(), 48)
        self.assertEqual(mesh.num_cells(), 70)

    def testUnitCube(self):
        """Create mesh of unit cube."""
        mesh = UnitCube(5, 7, 9)
        self.assertEqual(mesh.num_vertices(), 480)
        self.assertEqual(mesh.num_cells(), 1890)

class MeshRefinement(unittest.TestCase):

    def testRefineUnitSquare(self):
        """Refine mesh of unit square."""
        mesh = UnitSquare(5, 7)
        mesh = refine(mesh)
        self.assertEqual(mesh.num_vertices(), 165)
        self.assertEqual(mesh.num_cells(), 280)

    def testRefineUnitCube(self):
        """Refine mesh of unit cube."""
        mesh = UnitCube(5, 7, 9)
        mesh = refine(mesh)
        self.assertEqual(mesh.num_vertices(), 3135)
        self.assertEqual(mesh.num_cells(), 15120)

class MeshIterators(unittest.TestCase):

    def testVertexIterators(self):
        """Iterate over vertices."""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for v in vertices(mesh):
            n += 1
        self.assertEqual(n, mesh.num_vertices())

    def testEdgeIterators(self):
        """Iterate over edges."""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for e in edges(mesh):
            n += 1
        self.assertEqual(n, mesh.num_edges())

    def testFaceIterators(self):
        """Iterate over faces."""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for f in faces(mesh):
            n += 1
        self.assertEqual(n, mesh.num_faces())

    def testFacetIterators(self):
        """Iterate over facets."""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for f in facets(mesh):
            n += 1
        self.assertEqual(n, mesh.num_facets())

    def testCellIterators(self):
        """Iterate over cells."""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for c in cells(mesh):
            n += 1
        self.assertEqual(n, mesh.num_cells())

    def testMixedIterators(self):
        """Iterate over vertices of cells."""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for c in cells(mesh):
            for v in vertices(c):
                n += 1
        self.assertEqual(n, 4*mesh.num_cells())

class BoundaryExtraction(unittest.TestCase):

    def testBoundaryComputation(self):
        """Compute boundary of mesh."""
        mesh = UnitCube(2, 2, 2)
        boundary = BoundaryMesh(mesh)
        self.assertEqual(boundary.num_vertices(), 26)
        self.assertEqual(boundary.num_cells(), 48)

    def testBoundaryBoundary(self):
        """Compute boundary of boundary."""
        mesh = UnitCube(2, 2, 2)
        b0 = BoundaryMesh(mesh)
        b1 = BoundaryMesh(b0)
        self.assertEqual(b1.num_vertices(), 0)
        self.assertEqual(b1.num_cells(), 0)

class MeshFunctions(unittest.TestCase):

    def setUp(self):
        self.mesh = UnitSquare(3,3)
        self.f = MeshFunction('int', self.mesh, 0)

    def testAssign(self):
        """Assign value of mesh function."""
        f = self.f
        f[3] = 10
        v = Vertex(self.mesh, 3)
        self.assertEqual(f[v], 10)

    def testWrite(self):
        """Construct and save a simple meshfunction."""
        f = self.f
        f[0] = 1
        f[1] = 2
        file = File("saved_mesh_function.xml")
        file << f

    def testRead(self):
        """Construct and save a simple meshfunction. Then read it back from
        file."""
        mf = self.mesh.data().create_mesh_function("mesh_data_function", 2)
        mf[0] = 3
        mf[1] = 4

        self.f[0] =1
        self.f[1] =2
        file = File("saved_mesh_function.xml")
        file << self.f
        f = MeshFunction('int', self.mesh, "saved_mesh_function.xml")
        assert all(f.values() == self.f.values())

    def testSubsetIterators(self):
        def inside1(x):
            return x[0]<=0.5
        def inside2(x):
            return x[0]>=0.5
        sd1 = AutoSubDomain(inside1)
        sd2 = AutoSubDomain(inside2)
        cf = CellFunction('uint', self.mesh)
        cf.set_all(0)
        sd1.mark(cf, 1)
        sd2.mark(cf, 2)

        for i in range(3):
            num = 0
            for e in SubsetIterator(cf, i):
                num +=1
            self.assertEqual(num,6)

class NamedMeshFunctions(unittest.TestCase):

    def setUp(self):
        self.names = ["Cell", "Vertex", "Edge", "Face", "Facet"]
        self.tps = ['int', 'uint', 'bool', 'double']
        self.mesh = UnitCube(3,3,3)
        self.funcs = {}
        for tp in self.tps:
            for name in self.names:
                self.funcs[(tp, name)] = eval("%sFunction('%s', self.mesh)"%(name, tp))
            
    def test_size(self):
        for tp in self.tps:
            for name in self.names:
                if name is "Vertex":
                    self.assertEqual(self.funcs[(tp, name)].size(), self.mesh.num_vertices())
                else:
                    self.assertEqual(self.funcs[(tp, name)].size(),
                                     getattr(self.mesh, "num_%ss"%name.lower())())

    def test_access_type(self):
        type_dict = dict(int=int, uint=int, double=float, bool=bool)
        for tp in self.tps:
            for name in self.names:
                self.assertTrue(isinstance(self.funcs[(tp, name)][0], type_dict[tp]))

    def test_numpy_access(self):
        for tp in self.tps:
            for name in self.names:
                values = self.funcs[(tp, name)].values()
                values[:] = numpy.random.rand(len(values))
                self.assertTrue(all(values[i]==self.funcs[(tp, name)][i]
                                    for i in xrange(len(values))))

class InputOutput(unittest.TestCase):

    def testMeshXML2D(self):
        """Write and read 2D mesh to/from file"""
        mesh_out = UnitSquare(3, 3)
        mesh_in  = Mesh()
        file = File("unitsquare.xml")
        file << mesh_out
        file >> mesh_in
        self.assertEqual(mesh_in.num_vertices(), 16)

    def testMeshXML3D(self):
        """Write and read 3D mesh to/from file"""
        mesh_out = UnitCube(3, 3, 3)
        mesh_in  = Mesh()
        file = File("unitcube.xml")
        file << mesh_out
        file >> mesh_in
        self.assertEqual(mesh_in.num_vertices(), 64)

    def testMeshFunction(self):
        """Write and read mesh function to/from file"""
        mesh = UnitSquare(1, 1)
        f = MeshFunction('int', mesh, 0)
        f[0] = 2
        f[1] = 4
        f[2] = 6
        f[3] = 8
        file = File("meshfunction.xml")
        file << f
        g = MeshFunction('int', mesh, 0)
        file >> g
        for v in vertices(mesh):
            self.assertEqual(f[v], g[v])

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

class IntersectionOperator(unittest.TestCase):
    def testIntersectPoints(self):
        pass

    def testIntersectPoints(self):
        pass

    def testIntersectMesh2D(self):
        pass

    def testIntersectMesh3D(self):
        pass


if __name__ == "__main__":
    unittest.main()
