#!/usr/bin/env py.test

""" Tests for the meshconvert module."""

# Copyright (C) 2012
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
# Modified by Jan Blechta 2012
#
# First added:
# Last changed: 2014-05-30

from __future__ import print_function
import pytest
import os
import glob
import tempfile

from dolfin_utils.meshconvert import meshconvert
from dolfin_utils.meshconvert.meshconvert import DataHandler
from dolfin import MPI, mpi_comm_world
import six
from functools import reduce
from dolfin_utils.test import skip_in_parallel


class Tester:
    def assertTrue(self, a):
        assert a

    assert_ = assertTrue

    def assertFalse(self, a):
        assert not a

    def assertEqual(self, a, b):
        assert a == b

    def assertAlmostEqual(self, a, b):
        assert abs(a-b) < 1e-7

    def assertNotEqual(self, a, b):
        assert a != b

    def assertIsInstance(self, obj, cls):
        assert isinstance(obj, cls)

    def assertNotIsInstance(self, obj, cls):
        assert not isinstance(obj, cls)

    def assertRaises(self, e, f, *args):
        if args==[]:
            with pytest.raises(e):
                f()
        elif len(args)==1:
            with pytest.raises(e):
                f(args[0])
        elif len(args)==2:
            with pytest.raises(e):
                f(args[0], args[1])

    def assertEqualValues(self, A, B):
        B = as_ufl(B)
        self.assertEqual(A.ufl_shape, B.ufl_shape)
        self.assertEqual(inner(A-B, A-B)(None), 0)


class TestCase(Tester):
    def _get_tempfname(self, suffix=None):
        fd, fname = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        return fname

class _TestHandler(DataHandler):
    def __init__(self, cell_type, mesh_dim, test):
        DataHandler.__init__(self)
        self.vertices, self.cells, self.functions = [], [], {}
        self.vertices_ended = self.cells_ended = self.closed = False
        self.__type, self.__dim = cell_type, mesh_dim
        self.__test = test
        test.assertEqual(self._state, self.State_Invalid)
        self.test_warnings = []

    def set_mesh_type(self, *args):
        DataHandler.set_mesh_type(self, *args)
        test = self.__test
        test.assertEqual(self._state, self.State_Init)
        test.assertEqual(self._cell_type, self.__type)
        test.assertEqual(self._dim, self.__dim)

    def start_vertices(self, num_vertices):
        DataHandler.start_vertices(self, num_vertices)
        self.vertices = []
        for i in range(num_vertices):
            self.vertices.append(None)

    def add_vertex(self, vertex, coords):
        DataHandler.add_vertex(self, vertex, coords)
        self.vertices[vertex] = tuple(coords)

    def end_vertices(self):
        DataHandler.end_vertices(self)
        self.vertices_ended = True

    def start_cells(self, num_cells):
        DataHandler.start_cells(self, num_cells)
        for i in range(num_cells):
            self.cells.append(None)

    def add_cell(self, cell, nodes):
        DataHandler.add_cell(self, cell, nodes)
        self.cells[cell] = tuple(nodes)

    def end_cells(self):
        DataHandler.end_cells(self)
        self.cells_ended = True

    def start_meshfunction(self, name, dim, sz):
        DataHandler.start_meshfunction(self, name, dim, sz)
        entries = []
        for i in range(sz):
            entries.append(None)
        self.__curfunc = self.functions[name] = [dim, sz, entries, False]

    def add_entity_meshfunction(self, index, value):
        DataHandler.add_entity_meshfunction(self, index, value)
        self.__curfunc[2][index] = value

    def end_meshfunction(self):
        DataHandler.end_meshfunction(self)
        self.__curfunc[-1] = True

    def warn(self, msg):
        self.test_warnings.append(msg)

    def close(self):
        DataHandler.close(self)
        self.__test.assertEqual(self._state, self.State_Invalid)
        self.closed = True

class _ConverterTest(TestCase):
    """ Base converter test class.
    """

class AbaqusTest(_ConverterTest):
    """ Test AbaqusConverter."""

    def test_success(self):
        """ Test success case.
        """
        handler = self.__convert("abaqus.inp")
        # Verify vertices
        self.assertEqual(handler.vertices, [
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 10.0),
            (10.0, 10.0, 11.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 11.0),
            (10.0, 10.0, 10.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0)
            ])
        self.assert_(handler.vertices_ended)

        # Verify cells
        self.assertEqual(handler.cells, [
            (0, 9, 3, 8),
            (0, 9, 1, 2),
            (3, 8, 1, 2)
            ])
        self.assert_(handler.cells_ended)

        # Verify materials
        print(list(handler.functions.keys()))
        #self.assertEqual(handler.functions.keys(), ["material"])
        #dim, sz, entries, ended = handler.functions["material"]
        #self.assertEqual(dim, 3)
        #self.assertEqual(sz, 2)
        ## Cell 0 should have material 0, cell 1 material 1
        #self.assertEqual(entries, [0, 1])
        #self.assert_(ended)

        self.assert_(handler.closed)

    def xtest_error(self):
        """ Test various cases of erroneus input.
        """
        def convert(fname, text, error=False):
            f = file(fname, "w")
            f.write(text)
            f.close()
            if not error:
                handler = self.__convert(fname)
                self.assert_(handler.test_warnings)
            else:
                self.assertRaises(meshconvert.ParseError, self.__convert, fname)
            os.remove(fname)

        fname = self._get_tempfname(suffix=".inp")

        # Too few coordinates
#        convert(fname, """*NODE
#1, 0, 0""")
        # Non-numeric index
#        convert(fname, """*NODE
#a, 0, 0, 0""")
        # Non-numeric coordinate
#        convert(fname, """*NODE
#1, 0, 0, a""")
        # Unsupported element type, also the body should be ignored
#        convert(fname, """*ELEMENT, TYPE=sometype
#0
#""")
#        # Bad parameter syntax
#        convert(fname, "*ELEMENT, TYPE=sometype, BAD")
#        # Missing type specifier
#        convert(fname, "*ELEMENT", error=True)
#        # Non-existent node
#        convert(fname, """*NODE
#1, 0, 0, 0
#2, 0, 0, 0
#3, 0, 0, 0
#*ELEMENT, TYPE=C3D4
#1, 1, 2, 3, 4
#""", error=True)
#        # Too few nodes
#        convert(fname, """*NODE
#1, 0, 0, 0
##2, 0, 0, 0
#3, 0, 0, 0
#*ELEMENT, TYPE=C3D4
#1, 1, 2, 3
#""", error=True)
#        # Non-existent element set
#        convert(fname, """*MATERIAL, NAME=MAT
#*SOLID SECTION, ELSET=NONE, MATERIAL=MAT""", error=True)

    def test_facet_success(self):
        """ Test facet export.
        """
        dim = 3
        nb_facets = 1170  # The total number of facets in the mesh
        marker_counter = {0: 990,
                          1: 42,
                          2: 42,
                          3: 96,
                          4: 0}
        handler = self.__convert("abaqus_facet.inp")
        self.assert_(handler.vertices_ended)
        self.assert_(handler.cells_ended)

        self.assert_("facet_region" in list(handler.functions.keys()))
        cell_type = DataHandler.CellType_Triangle
        function_dim, sz, entries, ended = handler.functions["facet_region"]

        # the dimension of the meshfunction should be dim-1
        self.assertEqual(function_dim, dim - 1)
        # There should be size facets in the mesh function
        self.assertEqual(len(entries), nb_facets)
        self.assertEqual(sz, nb_facets)


        # Check that the right number of facets are marked
        for marker, count in six.iteritems(marker_counter):
            self.assert_(len([i for i in entries if i == marker]) == count)

        self.assert_(ended)
        self.assert_(handler.closed)

    def __convert(self, fname):
        handler = _TestHandler(DataHandler.CellType_Tetrahedron, 3, self)
        if not os.path.isabs(fname):
            fname = os.path.join("data", fname)
        meshconvert.convert(fname, handler)
        return handler

class TestGmsh(_ConverterTest):
    """ Test Gmsh convertor.
    """
    def test_success(self):
        """ Test success case.
        """
        handler = self.__convert("gmsh.msh")
        # Verify vertices
        self.assertEqual(handler.vertices, [
            (0,    0,    0),
            (1,    0,    1),
            (-0,    0.8,  0.6),
            (0.3,  0.8, -0.1),
            (0.6,  0.3, -0.4),
            (0.5,  0,    0.5),
            (0.5,  0.4,  0.8),
            (0.76, 0.26, 0.63),
            (0.53, 0.53, 0.26),
            (0.8,  0.15, 0.3)
            ])
        self.assert_(handler.vertices_ended)

        # Verify cells
        self.assertEqual(handler.cells, [
            (9, 5, 1, 7),
            (4, 8, 0, 9),
            (8, 5, 0, 9),
            (8, 5, 9, 7),
            (8, 3, 4, 0),
            (8, 2, 3, 0),
            (7, 5, 6, 8),
            (5, 2, 6, 8),
            (1, 7, 5, 6),
            (5, 2, 8, 0)
            ])
        self.assert_(handler.cells_ended)

        # Verify physical regions
        self.assertEqual(list(handler.functions.keys()), ["physical_region"])
        dim, sz, entries, ended = handler.functions["physical_region"]
        self.assertEqual(dim, 3)
        self.assertEqual(sz, 10)        # There are 10 cells
        # Cells 0 thru 4 should be in region 1000, cells 5 thru 9 in
        # region 2000
        self.assertEqual(entries, [1000]*5 + [2000]*5)
        self.assert_(ended)
        self.assert_(handler.closed)

    # FIXME: test disabled, see https://bitbucket.org/fenics-project/dolfin/issues/682
    def xtest_1D_facet_markings_2 (self):
        """
        Test to see if the 1D facet markings behave as expected.
        2 vertices marked
        """
        marked_facets = [0,2]
        self._facet_marker_driver(1, 2, marked_facets, 11)

    # FIXME: test disabled, see https://bitbucket.org/fenics-project/dolfin/issues/682
    def xtest_2D_facet_markings_1 (self):
        """
        Test to see if the 2D facet markings behave as expected.
        1 edge marked
        """
        marked_facets = [7]
        self._facet_marker_driver(2, 1, marked_facets, 8)

    # FIXME: test disabled, see https://bitbucket.org/fenics-project/dolfin/issues/682
    def xtest_2D_facet_markings_2 (self):
        """
        Test to see if the 2D facet markings behave as expected.
        2 edges marked
        """
        marked_facets = [2,5]
        self._facet_marker_driver(2, 2, marked_facets, 8)

    # FIXME: test disabled, see https://bitbucket.org/fenics-project/dolfin/issues/682
    def xtest_2D_facet_markings_3 (self):
        """
        Test to see if the 2D facet markings behave as expected.
        3 edges marked
        """
        marked_facets = [5,6,7]
        self._facet_marker_driver(2, 3, marked_facets, 8)

    # FIXME: test disabled, see https://bitbucket.org/fenics-project/dolfin/issues/682
    def xtest_2D_facet_markings_4 (self):
        """
        Test to see if the 2D facet markings behave as expected.
        4 edges marked
        """
        marked_facets = [2,5,6,7]
        self._facet_marker_driver(2, 4, marked_facets, 8)

    # FIXME: test disabled, see https://bitbucket.org/fenics-project/dolfin/issues/682
    def xtest_3D_facet_markings_1 (self):
        """
        Test the marking of 3D facets
        Unit cube, 1 Face marked
        """
#         [0, 0, 0, 999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 999, 0, 0, 0, 0, 0, 0, 0, 0, 999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 999,
# 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        marked_facets = [3, 15, 24, 39,]
        self._facet_marker_driver(3, 1, marked_facets, 60)

    def _facet_marker_driver (self, dim, id, marked_facets, size ):
        if dim == 1:
            cell_type = DataHandler.CellType_Interval
        elif dim == 2:
            cell_type = DataHandler.CellType_Triangle
        elif dim == 3:
            cell_type = DataHandler.CellType_Tetrahedron

        handler = self.__convert("gmsh_test_facet_regions_%dD_%d.msh" % (dim, id), cell_type, dim)

        free_facets = list(range(size))

        for i in marked_facets:
            free_facets.remove(i)

        function_dim, sz, entries, ended = handler.functions["facet_region"]

        # the dimension of the meshfunction should be dim-1
        self.assertEqual(function_dim, dim-1)
        # There should be size facets in the mesh function
        self.assertEqual(len(entries), size)
        self.assertEqual(sz, size)
        # marked
        self.assert_( all ( entries[i] == 999 for i in marked_facets ) )
        # all other edges should be zero
        self.assert_( all ( entries[i] == 0 for i in free_facets ) )

        self.assert_(ended)
        self.assert_(handler.closed)

    def __convert(self, fname, cell_type=DataHandler.CellType_Tetrahedron, dim=3):
        handler = _TestHandler(cell_type, dim, self)
        if not os.path.isabs(fname):
            fname = os.path.join(os.path.dirname(__file__), "data", fname)
        meshconvert.convert(fname, handler)
        return handler

@skip_in_parallel
class TestTriangle(Tester):

    def test_convert_triangle(self): # Disabled because it fails, see FIXME below

        # test no. 1
        from dolfin import Mesh, MPI, mpi_comm_world

        fname = os.path.join(os.path.dirname(__file__), "data", "triangle")
        dfname = fname+".xml"

        # Read triangle file and convert to a dolfin xml mesh file
        meshconvert.triangle2xml(fname, dfname)

        # Read in dolfin mesh and check number of cells and vertices
        mesh = Mesh(dfname)
        self.assertEqual(mesh.num_vertices(), 96)
        self.assertEqual(mesh.num_cells(), 159)

        # Clean up
        os.unlink(dfname)

        # test no. 2
        from dolfin import MPI, Mesh, MeshFunction, \
                           edges, Edge, faces, Face, \
                           SubsetIterator, facets, CellFunction, mpi_comm_world

        fname = os.path.join(os.path.dirname(__file__), "data", "test_Triangle_3")
        dfname = fname+".xml"
        dfname0 = fname+".attr0.xml"

        # Read triangle file and convert to a dolfin xml mesh file
        meshconvert.triangle2xml(fname, dfname)

        # Read in dolfin mesh and check number of cells and vertices
        mesh = Mesh(dfname)
        mesh.init()
        mfun = MeshFunction('double', mesh, dfname0)
        self.assertEqual(mesh.num_vertices(), 58)
        self.assertEqual(mesh.num_cells(), 58)

        # Create a size_t CellFunction and assign the values based on the
        # converted Meshfunction
        cf = CellFunction("size_t", mesh)
        cf.array()[mfun.array()==10.0] = 0
        cf.array()[mfun.array()==-10.0] = 1

        # Meassure total area of cells with 1 and 2 marker
        add = lambda x, y : x+y
        area0 = reduce(add, (Face(mesh, cell.index()).area() \
                             for cell in SubsetIterator(cf, 0)), 0.0)
        area1 = reduce(add, (Face(mesh, cell.index()).area() \
                             for cell in SubsetIterator(cf, 1)), 0.0)
        total_area = reduce(add, (face.area() for face in faces(mesh)), 0.0)

        # Check that all cells in the two domains are either above or below y=0
        self.assertTrue(all(cell.midpoint().y()<0 for cell in SubsetIterator(cf, 0)))
        self.assertTrue(all(cell.midpoint().y()>0 for cell in SubsetIterator(cf, 1)))

        # Check that the areas add up
        self.assertAlmostEqual(area0+area1, total_area)

        # Measure the edge length of the two edge domains
        #edge_markers = mesh.domains().facet_domains()
        edge_markers = mesh.domains().markers(mesh.topology().dim()-1)
        self.assertTrue(edge_markers is not None)
        #length0 = reduce(add, (Edge(mesh, e.index()).length() \
        #                    for e in SubsetIterator(edge_markers, 0)), 0.0)
        length0, length1 = 0.0, 0.0
        for item in list(edge_markers.items()):
            if item[1] == 0:
                e = Edge(mesh, int(item[0]))
                length0 +=  Edge(mesh, int(item[0])).length()
            elif item [1] == 1:
                length1 +=  Edge(mesh, int(item[0])).length()

        # Total length of all edges and total length of boundary edges
        total_length = reduce(add, (e.length() for e in edges(mesh)), 0.0)
        boundary_length = reduce(add, (Edge(mesh, f.index()).length() \
                          for f in facets(mesh) if f.exterior()), 0.0)

        # Check that the edges add up
        self.assertAlmostEqual(length0 + length1, total_length)
        self.assertAlmostEqual(length1, boundary_length)

        # Clean up
        os.unlink(dfname)
        os.unlink(dfname0)

@skip_in_parallel
class TestDiffPack(Tester):
    def test_convert_diffpack(self):

        from dolfin import Mesh, MPI, MeshFunction, mpi_comm_world

        fname = os.path.join(os.path.dirname(__file__), "data", "diffpack_tet")
        dfname = fname+".xml"

        # Read triangle file and convert to a dolfin xml mesh file
        meshconvert.diffpack2xml(fname+".grid", dfname)

        # Read in dolfin mesh and check number of cells and vertices
        mesh = Mesh(dfname)
        self.assertEqual(mesh.num_vertices(), 27)
        self.assertEqual(mesh.num_cells(), 48)
        self.assertEqual(len(mesh.domains().markers(3)), 48)
        self.assertEqual(len(mesh.domains().markers(2)), 16)

        mf_basename = dfname.replace(".xml", "_marker_%d.xml")
        for marker, num in [(3, 9), (6, 9), (7, 3), (8, 1)]:

            mf_name = mf_basename % marker
            mf = MeshFunction("size_t", mesh, mf_name)
            self.assertEqual(sum(mf.array()==marker), num)
            os.unlink(mf_name)

        # Clean up
        os.unlink(dfname)

    def test_convert_diffpack_2d(self):

        from dolfin import Mesh, MPI, MeshFunction, mpi_comm_world

        fname = os.path.join(os.path.dirname(__file__), "data", "diffpack_tri")
        dfname = fname+".xml"

        # Read triangle file and convert to a dolfin xml mesh file
        meshconvert.diffpack2xml(fname+".grid", dfname)

        # Read in dolfin mesh and check number of cells and vertices
        mesh = Mesh(dfname)

        self.assertEqual(mesh.num_vertices(), 41)
        self.assertEqual(mesh.num_cells(), 64)
        self.assertEqual(len(mesh.domains().markers(2)), 64)

        mf_basename = dfname.replace(".xml", "_marker_%d.xml")
        for marker, num in [(1,10), (2,5), (3,5)]:

            mf_name = mf_basename % marker
            mf = MeshFunction("size_t", mesh, mf_name)
            self.assertEqual(sum(mf.array()==marker), num)
            os.unlink(mf_name)

        # Clean up
        os.unlink(dfname)
