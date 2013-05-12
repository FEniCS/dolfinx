"Unit tests for the mesh library"

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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Benjamin Kehlet 2012
# Modified by Marie E. Rognes 2012
# Modified by Johannes Ring 2013
# Modified by Jan Blechta 2013
#
# First added:  2006-08-08
# Last changed: 2013-02-22

import unittest
import numpy
from dolfin import *

class MeshConstruction(unittest.TestCase):

    def setUp(self):
        if MPI.num_processes() == 1:
            self.interval = UnitIntervalMesh(10)
        self.circle = UnitCircleMesh(5)
        self.square = UnitSquareMesh(5, 5)
        self.rectangle = RectangleMesh(0, 0, 2, 2, 5, 5)
        self.cube = UnitCubeMesh(3, 3, 3)
        self.box = BoxMesh(0, 0, 0, 2, 2, 2, 2, 2, 5)

    def testUFLCell(self):
        import ufl
        if MPI.num_processes() == 1:
            self.assertEqual(ufl.interval, self.interval.ufl_cell())
        self.assertEqual(ufl.triangle, self.circle.ufl_cell())
        self.assertEqual(ufl.triangle, self.square.ufl_cell())
        self.assertEqual(ufl.triangle, self.rectangle.ufl_cell())
        self.assertEqual(ufl.tetrahedron, self.cube.ufl_cell())
        self.assertEqual(ufl.tetrahedron, self.box.ufl_cell())

if MPI.num_processes() == 1:
    class SimpleShapes(unittest.TestCase):

        def testUnitSquareMesh(self):
            """Create mesh of unit square."""
            mesh = UnitSquareMesh(5, 7)
            self.assertEqual(mesh.num_vertices(), 48)
            self.assertEqual(mesh.num_cells(), 70)

        def testUnitCubeMesh(self):
            """Create mesh of unit cube."""
            mesh = UnitCubeMesh(5, 7, 9)
            self.assertEqual(mesh.num_vertices(), 480)
            self.assertEqual(mesh.num_cells(), 1890)

class MeshRefinement(unittest.TestCase):

    def testRefineUnitSquareMesh(self):
        """Refine mesh of unit square."""
        mesh = UnitSquareMesh(5, 7)
        mesh = refine(mesh)
        self.assertEqual(mesh.size_global(0), 165)
        self.assertEqual(mesh.size_global(2), 280)

    def testRefineUnitCubeMesh(self):
        """Refine mesh of unit cube."""
        mesh = UnitCubeMesh(5, 7, 9)
        mesh = refine(mesh)
        self.assertEqual(mesh.size_global(0), 3135)
        self.assertEqual(mesh.size_global(3), 15120)

# This test does not work in parallel because BoundaryMesh does not
# compute distributed mesh data
if MPI.num_processes() == 1:
    class BoundaryExtraction(unittest.TestCase):

        def testBoundaryComputation(self):
            """Compute boundary of mesh."""
            mesh = UnitCubeMesh(2, 2, 2)
            boundary = BoundaryMesh(mesh, "exterior")
            self.assertEqual(boundary.num_vertices(), 26)
            self.assertEqual(boundary.num_cells(), 48)

            def testBoundaryBoundary(self):
                """Compute boundary of boundary."""
                mesh = UnitCubeMesh(2, 2, 2)
                b0 = BoundaryMesh(mesh, "exterior")
                b1 = BoundaryMesh(b0, "exterior")
                self.assertEqual(b1.num_vertices(), 0)
                self.assertEqual(b1.num_cells(), 0)

if MPI.num_processes() == 1:
    class MeshFunctions(unittest.TestCase):

        def setUp(self):
            self.mesh = UnitSquareMesh(3, 3)
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
            cf = CellFunction('size_t', self.mesh)
            cf.set_all(0)
            sd1.mark(cf, 1)
            sd2.mark(cf, 2)

            for i in range(3):
                num = 0
                for e in SubsetIterator(cf, i):
                    num += 1
                self.assertEqual(num, 6)


# FIXME: Mesh IO tests should be in io test directory
if MPI.num_processes() == 1:
    class InputOutput(unittest.TestCase):

        def testMeshXML2D(self):
            """Write and read 2D mesh to/from file"""
            mesh_out = UnitSquareMesh(3, 3)
            mesh_in  = Mesh()
            file = File("unitsquare.xml")
            file << mesh_out
            file >> mesh_in
            self.assertEqual(mesh_in.num_vertices(), 16)

        def testMeshXML3D(self):
            """Write and read 3D mesh to/from file"""
            mesh_out = UnitCubeMesh(3, 3, 3)
            mesh_in  = Mesh()
            file = File("unitcube.xml")
            file << mesh_out
            file >> mesh_in
            self.assertEqual(mesh_in.num_vertices(), 64)

        def testMeshFunction(self):
            """Write and read mesh function to/from file"""
            mesh = UnitSquareMesh(1, 1)
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
        mesh = UnitSquareMesh(5, 5)
        self.assertEqual(mesh.geometry().dim(), 2)

    if MPI.num_processes() > 0 :
        def xtestGetCoordinates(self):
            """Get coordinates of vertices"""
            mesh = UnitSquareMesh(5, 5)
            self.assertEqual(len(mesh.coordinates()), 36)

    def testGetCells(self):
        """Get cells of mesh"""
        mesh = UnitSquareMesh(5, 5)
        self.assertEqual(MPI.sum(len(mesh.cells())), 50)


if MPI.num_processes() == 1:
    class IntersectionOperator(unittest.TestCase):
        def testIntersectPoint(self):
            from numpy import linspace
            mesh = UnitSquareMesh(10, 10)
            points = [Point(i+.05,.05) for i in linspace(-.4,1.4,19)]
            for p in points:
                if p.x()<0 or p.x()>1:
                    self.assertTrue(not mesh.intersected_cells(p))
                else:
                    self.assertTrue(mesh.intersected_cells(p))

        def testIntersectPoints(self):
            from numpy import linspace
            mesh = UnitSquareMesh(10, 10)
            points = [Point(i+.05,.05) for i in linspace(-.4,1.4,19)]
            all_intersected_entities = []
            for p in points:
                all_intersected_entities.extend(mesh.intersected_cells(p))
            for i0, i1 in zip(sorted(all_intersected_entities),
                              sorted(mesh.intersected_cells(points))):
                self.assertEqual(i0, i1)

        def testIntersectMesh2D(self):
            pass

        def testIntersectMesh3D(self):
            pass

        def testIntersectedCellWithSingleCellMesh(self):

            # 2D
            mesh = UnitTriangleMesh()

            # Point Intersection
            point = Point(0.3, 0.3)
            id = mesh.intersected_cell(point)
            self.assertEqual(id, 0)
            cells = mesh.intersected_cells([point, point, point])
            self.assertEqual(len(cells), 1)
            self.assertEqual(cells[0], 0)

            # Entity intersection
            v = Vertex(mesh, 0)
            id = mesh.intersected_cells(v)
            self.assertEqual(id, 0)

            # No intersection
            point = Point(1.2, 1.2)
            id = mesh.intersected_cell(point)
            self.assertEqual(id, -1)
            cells = mesh.intersected_cells([point, point, point])
            self.assertEqual(len(cells), 0)

            # 3D
            mesh = UnitTetrahedronMesh()

            # Point intersection
            point = Point(0.3, 0.3, 0.3)
            id = mesh.intersected_cell(point)
            self.assertEqual(id, 0)
            cells = mesh.intersected_cells([point, point, point])
            self.assertEqual(len(cells), 1)
            self.assertEqual(cells[0], 0)

            # Entity intersection
            v = Vertex(mesh, 0)
            id = mesh.intersected_cells(v)
            self.assertEqual(id, 0)

            # No intersection
            point = Point(1.2, 1.2, 1.2)
            id = mesh.intersected_cell(point)
            self.assertEqual(id, -1)
            cells = mesh.intersected_cells([point, point, point])
            self.assertEqual(len(cells), 0)

        def testClosestCellWithSingleCellMesh(self):
            mesh = UnitTriangleMesh()

            point = Point(0.3, 0.3)
            id = mesh.closest_cell(point)
            self.assertEqual(id, 0)

            point = Point(1.2, 1.2)
            id = mesh.closest_cell(point)
            self.assertEqual(id, 0)

            mesh = UnitTetrahedronMesh()

            point = Point(0.3, 0.3, 0.3)
            id = mesh.closest_cell(point)
            self.assertEqual(id, 0)

            point = Point(1.2, 1.2, 1.2)
            id = mesh.closest_cell(point)
            self.assertEqual(id, 0)


if MPI.num_processes() == 1:
    class CellRadii(unittest.TestCase):

        def setUp(self):
            # Create 1D mesh with degenerate cell
            self.mesh1d = UnitIntervalMesh(4)
            self.mesh1d.coordinates()[4] = self.mesh1d.coordinates()[3]

            # Create 2D mesh with one equilateral triangle
            self.mesh2d = UnitSquareMesh(1, 1, 'left')
            self.mesh2d.coordinates()[3] += 0.5*(sqrt(3.0)-1.0)

            # Create 3D mesh with regular tetrahedron and degenerate cells
            self.mesh3d = UnitCubeMesh(1, 1, 1)
            self.mesh3d.coordinates()[2][0] = 1.0
            self.mesh3d.coordinates()[7][1] = 0.0
            # Original tetrahedron from UnitCubeMesh(1, 1, 1)
            self.c0 = Cell(self.mesh3d, 0)
            # Degenerate cell
            self.c1 = Cell(self.mesh3d, 1)
            # Regular tetrahedron with edge sqrt(2)
            self.c5 = Cell(self.mesh3d, 5)

        def test_cell_inradius(self):
            self.assertAlmostEqual(self.c0.inradius(), (3.0-sqrt(3.0))/6.0)
            self.assertAlmostEqual(self.c1.inradius(), 0.0)
            self.assertAlmostEqual(self.c5.inradius(), sqrt(3.0)/6.0)

        def test_cell_diameter(self):
            from math import isnan
            self.assertAlmostEqual(self.c0.diameter(), sqrt(3.0))
            # Implementation of diameter() does not work accurately
            # for degenerate cells - sometimes yields NaN
            self.assertTrue(isnan(self.c1.diameter()))
            self.assertAlmostEqual(self.c5.diameter(), sqrt(3.0))

        def test_cell_radius_ratio(self):
            self.assertAlmostEqual(self.c0.radius_ratio(), sqrt(3.0)-1.0)
            self.assertAlmostEqual(self.c1.radius_ratio(), 0.0)
            self.assertAlmostEqual(self.c5.radius_ratio(), 1.0)

        def test_hmin_hmax(self):
            self.assertAlmostEqual(self.mesh1d.hmin(), 0.0)
            self.assertAlmostEqual(self.mesh1d.hmax(), 0.25)
            self.assertAlmostEqual(self.mesh2d.hmin(), sqrt(2.0))
            self.assertAlmostEqual(self.mesh2d.hmax(), 2.0*sqrt(6.0)/3.0)
            # nans are not taken into account in hmax and hmin
            self.assertAlmostEqual(self.mesh3d.hmin(), sqrt(3.0))
            self.assertAlmostEqual(self.mesh3d.hmax(), sqrt(3.0))

        def test_rmin_rmax(self):
            self.assertAlmostEqual(self.mesh1d.rmin(), 0.0)
            self.assertAlmostEqual(self.mesh1d.rmax(), 0.125)
            self.assertAlmostEqual(self.mesh2d.rmin(), 1.0/(2.0+sqrt(2.0)))
            self.assertAlmostEqual(self.mesh2d.rmax(), sqrt(6.0)/6.0)
            self.assertAlmostEqual(self.mesh3d.rmin(), 0.0)
            self.assertAlmostEqual(self.mesh3d.rmax(), sqrt(3.0)/6.0)

        def test_radius_ratio_min_radius_ratio_max(self):
            self.assertAlmostEqual(self.mesh1d.radius_ratio_min(), 0.0)
            self.assertAlmostEqual(self.mesh1d.radius_ratio_max(), 1.0)
            self.assertAlmostEqual(self.mesh2d.radius_ratio_min(),
                                     2.0*sqrt(2.0)/(2.0+sqrt(2.0)) )
            self.assertAlmostEqual(self.mesh2d.radius_ratio_max(), 1.0)
            self.assertAlmostEqual(self.mesh3d.radius_ratio_min(), 0.0)
            self.assertAlmostEqual(self.mesh3d.radius_ratio_max(), 1.0)


class MeshOrientations(unittest.TestCase):

    def setUp(self):
        pass

    def test_basic_cell_orientations(self):
        "Test that default cell orientations initialize and update as expected."
        mesh = UnitIntervalMesh(6)
        orientations = mesh.cell_orientations()
        self.assertEqual(len(orientations), mesh.num_cells())
        for i in range(mesh.num_cells()):
            self.assertEqual(orientations[i], -1)

        orientations[0] = 1
        self.assertEqual(mesh.cell_orientations()[0], 1)

    def test_cell_orientations(self):
        "Test that cell orientations update as expected."
        mesh = UnitIntervalMesh(6)
        mesh.init_cell_orientations(Expression(("0.0", "1.0", "0.0")))
        for i in range(mesh.num_cells()):
            self.assertEqual(mesh.cell_orientations()[i], 0)

        mesh = UnitSquareMesh(2, 2)
        mesh.init_cell_orientations(Expression(("0.0", "0.0", "1.0")))
        reference = numpy.array((0, 1, 0, 1, 0, 1, 0, 1))
        # Only compare against reference in serial (don't know how to
        # compare in parallel)
        if MPI.num_processes() == 1:
            for i in range(mesh.num_cells()):
                self.assertEqual(mesh.cell_orientations()[i], reference[i])

        mesh = BoundaryMesh(UnitSquareMesh(2, 2), "exterior")
        mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))
        print mesh.cell_orientations()

if __name__ == "__main__":
    unittest.main()
