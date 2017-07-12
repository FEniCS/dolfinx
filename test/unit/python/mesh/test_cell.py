#!/usr/bin/env py.test

"""Unit tests for the Cell class"""

# Copyright (C) 2013 Anders Logg
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
# First added:  2013-04-18
# Last changed: 2014-05-30

import pytest
import numpy
from dolfin import *
import FIAT
import six

from dolfin_utils.test import skip_in_parallel, skip_in_release


@skip_in_parallel
def test_distance_interval():

    mesh = UnitIntervalMesh.create(1)
    cell = Cell(mesh, 0)

    assert round(cell.distance(Point(-1.0)) - 1.0, 7) == 0
    assert round(cell.distance(Point(0.5)) - 0.0, 7) == 0


@skip_in_parallel
def test_distance_triangle():

    mesh = UnitSquareMesh(1, 1)
    cell = Cell(mesh, 1)

    assert round(cell.distance(Point(-1.0, -1.0)) - numpy.sqrt(2), 7) == 0
    assert round(cell.distance(Point(-1.0, 0.5)) - 1, 7) == 0
    assert round(cell.distance(Point(0.5, 0.5)) - 0.0, 7) == 0


@skip_in_parallel
def test_distance_tetrahedron():

    mesh = UnitCubeMesh(1, 1, 1)
    cell = Cell(mesh, 5)

    assert round(cell.distance(Point(-1.0, -1.0, -1.0))-numpy.sqrt(3), 7) == 0
    assert round(cell.distance(Point(-1.0, 0.5, 0.5)) - 1, 7) == 0
    assert round(cell.distance(Point(0.5, 0.5, 0.5)) - 0.0, 7) == 0


@skip_in_release
@skip_in_parallel
def test_issue_568():
    mesh = UnitSquareMesh(4, 4)
    cell = Cell(mesh, 0)

    # Should throw an error, not just segfault (only works in DEBUG mode!)
    with pytest.raises(RuntimeError):
        cell.facet_area(0)

    # Should work after initializing the connectivity
    mesh.init(2, 1)
    cell.facet_area(0)


def test_volume_quadrilateralR2():

    mesh = UnitQuadMesh.create(mpi_comm_self(), 1, 1)
    cell = Cell(mesh, 0)

    assert cell.volume() == 1.0


@pytest.mark.parametrize('coordinates', [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], 
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])
def test_volume_quadrilateralR3(coordinates):

    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "quadrilateral", 2, 3)
    editor.init_vertices(4)
    editor.init_cells(1)
    editor.add_vertex(0, Point(numpy.array(coordinates[0])))
    editor.add_vertex(1, Point(numpy.array(coordinates[1])))
    editor.add_vertex(2, Point(numpy.array(coordinates[2])))
    editor.add_vertex(3, Point(numpy.array(coordinates[3])))
    editor.add_cell(0,numpy.array([0, 1, 2, 3],dtype=numpy.uintp))
    editor.close()
    mesh.init()
    cell = Cell(mesh, 0)

    assert cell.volume() == 1.0


@pytest.mark.parametrize('scaling', [1e0, 1e-5, 1e-10, 1e-15, 1e-20, 1e-30,
    1e5, 1e10, 1e15, 1e20, 1e30])
def test_volume_quadrilateral_coplanarity_check_1(scaling):

    with pytest.raises(RuntimeError) as error:
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, "quadrilateral", 2, 3)
        editor.init_vertices(4)
        editor.init_cells(1)
        # Unit square cell scaled down by 'scaling' and the first vertex is distorted so that the vertices are clearly non coplanar
        editor.add_vertex(0, Point(scaling, 0.5 * scaling, 0.6 * scaling))
        editor.add_vertex(1, Point(0.0, scaling, 0.0))
        editor.add_vertex(2, Point(0.0, 0.0, scaling))
        editor.add_vertex(3, Point(0.0, scaling, scaling))
        editor.add_cell(0,numpy.array([0, 1, 2, 3],dtype=numpy.uintp))
        editor.close()
        mesh.init()
        cell = Cell(mesh, 0)
        volume = cell.volume()

    assert "are not coplanar" in str(error.value)


# Test when |p0-p3| is ~ 1 but |p1-p2| is small
# The cell is degenerate when scale is below 1e-17, it is expected to fail the test.
@pytest.mark.parametrize('scaling', [1e0, 1e-5, 1e-10, 1e-15])
def test_volume_quadrilateral_coplanarity_check_2(scaling):

    with pytest.raises(RuntimeError) as error:
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, "quadrilateral", 2, 3)
        editor.init_vertices(4)
        editor.init_cells(1)
        # Unit square cell scaled down by 'scaling' and the first vertex is distorted so that the vertices are clearly non coplanar
        editor.add_vertex(0, Point(1.0, 0.5, 0.6))
        editor.add_vertex(1, Point(0.0, scaling, 0.0))
        editor.add_vertex(2, Point(0.0, 0.0, scaling))
        editor.add_vertex(3, Point(0.0, 1.0, 1.0))
        editor.add_cell(0,numpy.array([0, 1, 2, 3],dtype=numpy.uintp))
        editor.close()
        mesh.init()
        cell = Cell(mesh, 0)
        volume = cell.volume()

    assert "are not coplanar" in str(error.value)


@pytest.mark.parametrize('mesh_factory', [
    (UnitIntervalMesh, (8,)),
    (UnitSquareMesh, (4, 4)),
    (UnitCubeMesh, (2, 2, 2)),
    (UnitQuadMesh.create, (4, 4)),
    (UnitHexMesh.create, (2, 2, 2)),
])
def test_cell_topology_against_fiat(mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)
    assert mesh.ordered()

    # Create DOLFIN and FIAT cell
    cell_name = CellType.type2string(mesh.type().cell_type())
    fiat_cell = FIAT.ufc_cell(cell_name)

    # Initialize mesh entities
    tdim = mesh.topology().dim()
    for d in six.moves.range(tdim+1):
        mesh.init(d)

    # Test topology
    for cell in cells(mesh):
        vertex_global_indices = cell.entities(0)

        for d, d_topology in six.iteritems(fiat_cell.get_topology()):
            entities = cell.entities(d)

            # Fixup for highest dimension
            if len(entities) == 0:
                entities = (cell.index(),)

            for entity_index, entity_topology in six.iteritems(d_topology):

                # Check that vertices of mesh entity match FIAT topology
                entity = MeshEntity(mesh, d, entities[entity_index])
                entity_vertices = entity.entities(0)
                assert all(vertex_global_indices[numpy.array(entity_topology)]
                           == entity_vertices)
