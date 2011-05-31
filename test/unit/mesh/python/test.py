"""Unit tests for the mesh library"""

# Copyright (C) 2006 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2006-08-08
# Last changed: 2011-03-23

import unittest
import numpy
from dolfin import *

class MeshConstruction(unittest.TestCase):

    def setUp(self):
        self.intervall = UnitInterval(10)
        self.circle = UnitCircle(5)
        self.square = UnitSquare(5,5)
        self.rectangle = Rectangle(0,0,2,2,5,5)
        self.cube = UnitCube(3,3,3)
        self.sphere = UnitSphere(5)
        self.box = Box(0,0,0,2,2,2,2,2,5)
    
    def testUFLCell(self):
        import ufl
        self.assertEqual(ufl.interval, self.intervall.ufl_cell())
        self.assertEqual(ufl.triangle, self.circle.ufl_cell())
        self.assertEqual(ufl.triangle, self.square.ufl_cell())
        self.assertEqual(ufl.triangle, self.rectangle.ufl_cell())
        self.assertEqual(ufl.tetrahedron, self.cube.ufl_cell())
        self.assertEqual(ufl.tetrahedron, self.sphere.ufl_cell())
        self.assertEqual(ufl.tetrahedron, self.box.ufl_cell())

class MeshEditorTest(unittest.TestCase):

    def testTriangle(self):
        # Create mesh object and open editor
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 2, 2)
        editor.init_vertices(3)
        editor.init_cells(1)

        # Add vertices
        p = Point(0.0, 0.0)
        editor.add_vertex(0, p)
        p = Point(1.0, 0.0)
        editor.add_vertex(1, p)
        p = Point(0.0, 1.0)
        editor.add_vertex(2, p)

        # Add cell
        editor.add_cell(0, 0, 1, 2)

        # Close editor
        editor.close()

class MeshIterators(unittest.TestCase):

    def testVertexIterators(self):
        """Iterate over vertices."""
        mesh = UnitCube(5, 5, 5)

        # Test connectivity
        cons = [(i, mesh.topology()(0,i)) for i in xrange(4)]

        # Test writability
        for i, con in cons:
            def assign(con, i):
                con(i)[0] = 1
            self.assertRaises(RuntimeError, assign, con, i)
        
        n = 0
        for i, v in enumerate(vertices(mesh)):
            n += 1
            for j, con in cons:
                self.assertTrue(numpy.all(con(i) == v.entities(j)))
        
        self.assertEqual(n, mesh.num_vertices())
        
        # Check coordinate assignment
        # FIXME: Outcomment to hopefully please Mac-buildbot
        #end_point = numpy.array([v.x(0), v.x(1), v.x(2)])
        #mesh.coordinates()[:] += 2
        #self.assertEqual(end_point[0] + 2, mesh.coordinates()[-1,0])
        #self.assertEqual(end_point[1] + 2, mesh.coordinates()[-1,1])
        #self.assertEqual(end_point[2] + 2, mesh.coordinates()[-1,2])

    def testEdgeIterators(self):
        """Iterate over edges."""
        mesh = UnitCube(5, 5, 5)

        # Test connectivity
        cons = [(i, mesh.topology()(1,i)) for i in xrange(4)]
        
        # Test writability
        for i, con in cons:
            def assign(con, i):
                con(i)[0] = 1
            self.assertRaises(RuntimeError, assign, con, i)
        
        n = 0
        for i, e in enumerate(edges(mesh)):
            n += 1
            for j, con in cons:
                self.assertTrue(numpy.all(con(i) == e.entities(j)))
        
        self.assertEqual(n, mesh.num_edges())

    def testFaceIterators(self):
        """Iterate over faces."""
        mesh = UnitCube(5, 5, 5)

        # Test connectivity
        cons = [(i, mesh.topology()(2,i)) for i in xrange(4)]
        
        # Test writability
        for i, con in cons:
            def assign(con, i):
                con(i)[0] = 1
            self.assertRaises(RuntimeError, assign, con, i)
        
        n = 0
        for i, f in enumerate(faces(mesh)):
            n += 1
            for j, con in cons:
                self.assertTrue(numpy.all(con(i) == f.entities(j)))
        
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

        # Test connectivity
        cons = [(i, mesh.topology()(3,i)) for i in xrange(4)]
        
        # Test writability
        for i, con in cons:
            def assign(con, i):
                con(i)[0] = 1
            self.assertRaises(RuntimeError, assign, con, i)
        
        n = 0
        for i, c in enumerate(cells(mesh)):
            n += 1
            for j, con in cons:
                self.assertTrue(numpy.all(con(i) == c.entities(j)))
        
        self.assertEqual(n, mesh.num_cells())

    def testMixedIterators(self):
        """Iterate over vertices of cells."""
        mesh = UnitCube(5, 5, 5)
        n = 0
        for c in cells(mesh):
            for v in vertices(c):
                n += 1
        self.assertEqual(n, 4*mesh.num_cells())

# FIXME: The following test breaks in parallel
if MPI.num_processes() == 1:
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
            self.mesh = UnitSquare(3, 3)
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
            #mf = self.mesh.data().create_mesh_function("mesh_data_function", 2)
            #print "***************", mf
            #mf[0] = 3
            #mf[1] = 4

            #self.f[0] = 1
            #self.f[1] = 2
            #file = File("saved_mesh_function.xml")
            #file << self.f
            #f = MeshFunction('int', self.mesh, "saved_mesh_function.xml")
            #assert all(f.array() == self.f.array())

        def testSubsetIterators(self):
            def inside1(x):
                return x[0] <= 0.5
            def inside2(x):
                return x[0] >= 0.5
            sd1 = AutoSubDomain(inside1)
            sd2 = AutoSubDomain(inside2)
            cf = CellFunction('int', self.mesh)
            cf.set_all(0)
            sd1.mark(cf, 1)
            sd2.mark(cf, 2)

            #for i in range(3):
            #    num = 0
            #    for e in SubsetIterator(cf, i):
            #        num += 1
            #    self.assertEqual(num, 6)


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
        def testIntersectPoint(self):
            from numpy import linspace
            mesh = UnitSquare(10, 10)
            points = [Point(i+.05,.05) for i in linspace(-.4,1.4,19)]
            for p in points:
                if p.x()<0 or p.x()>1:
                    self.assertTrue(not mesh.all_intersected_entities(p))
                else:
                    self.assertTrue(mesh.all_intersected_entities(p))


        def testIntersectPoints(self):
            from numpy import linspace
            mesh = UnitSquare(10, 10)
            points = [Point(i+.05,.05) for i in linspace(-.4,1.4,19)]
            all_intersected_entities = []
            for p in points:
                all_intersected_entities.extend(mesh.all_intersected_entities(p))
            for i0, i1 in zip(sorted(all_intersected_entities),
                              sorted(mesh.all_intersected_entities(points))):
                self.assertEqual(i0, i1)

        def testIntersectMesh2D(self):
            pass

        def testIntersectMesh3D(self):
            pass


if __name__ == "__main__":
    unittest.main()
