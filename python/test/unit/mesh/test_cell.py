# Copyright (C) 2013 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import pytest
from dolfinx_utils.test.skips import skip_in_parallel

from dolfinx import (MPI, Mesh, MeshEntity, UnitCubeMesh, UnitIntervalMesh,
                    UnitSquareMesh, cpp)
from dolfinx.cpp.mesh import CellType


@skip_in_parallel
def test_distance_interval():
    mesh = UnitIntervalMesh(MPI.comm_self, 1)
    cell = MeshEntity(mesh, mesh.topology.dim, 0)
    assert cpp.geometry.squared_distance(cell, numpy.array([-1.0, 0, 0])) == pytest.approx(1.0)
    assert cpp.geometry.squared_distance(cell, numpy.array([0.5, 0, 0])) == pytest.approx(0.0)


@skip_in_parallel
def test_distance_triangle():
    mesh = UnitSquareMesh(MPI.comm_self, 1, 1)
    cell = MeshEntity(mesh, mesh.topology.dim, 1)
    assert cpp.geometry.squared_distance(cell, numpy.array([-1.0, -1.0, 0.0])) == pytest.approx(2.0)
    assert cpp.geometry.squared_distance(cell, numpy.array([-1.0, 0.5, 0.0])) == pytest.approx(1.0)
    assert cpp.geometry.squared_distance(cell, numpy.array([0.5, 0.5, 0.0])) == pytest.approx(0.0)


@skip_in_parallel
def test_distance_tetrahedron():
    mesh = UnitCubeMesh(MPI.comm_self, 1, 1, 1)
    cell = MeshEntity(mesh, mesh.topology.dim, 5)
    assert cpp.geometry.squared_distance(cell, numpy.array([-1.0, -1.0, -1.0])) == pytest.approx(3.0)
    assert cpp.geometry.squared_distance(cell, numpy.array([-1.0, 0.5, 0.5])) == pytest.approx(1.0)
    assert cpp.geometry.squared_distance(cell, numpy.array([0.5, 0.5, 0.5])) == pytest.approx(0.0)


@pytest.mark.parametrize(
    'mesh', [UnitIntervalMesh(MPI.comm_world, 8),
             UnitSquareMesh(MPI.comm_world, 8, 9, CellType.triangle),
             UnitSquareMesh(MPI.comm_world, 8, 9, CellType.quadrilateral),
             UnitCubeMesh(MPI.comm_world, 8, 9, 5, CellType.tetrahedron)])
def test_volume_cells(mesh):
    num_cells = mesh.num_entities(mesh.topology.dim)
    v = cpp.mesh.volume_entities(mesh, range(num_cells), mesh.topology.dim)
    v = MPI.sum(mesh.mpi_comm(), v.sum())
    assert v == pytest.approx(1.0, rel=1e-9)


def test_volume_quadrilateralR2():
    mesh = UnitSquareMesh(MPI.comm_self, 1, 1, CellType.quadrilateral)
    assert cpp.mesh.volume_entities(mesh, [0], mesh.topology.dim) == 1.0


@pytest.mark.parametrize(
    'coordinates',
    [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
     [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]])
def test_volume_quadrilateralR3(coordinates):
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral,
                numpy.array(coordinates, dtype=numpy.float64),
                numpy.array([[0, 1, 2, 3]], dtype=numpy.int32), [],
                cpp.mesh.GhostMode.none)
    mesh.create_connectivity_all()
    assert cpp.mesh.volume_entities(mesh, [0], mesh.topology.dim) == 1.0


@pytest.mark.parametrize(
    'scaling',
    [1e0, 1e-5, 1e-10, 1e-15, 1e-20, 1e-30, 1e5, 1e10, 1e15, 1e20, 1e30])
def test_volume_quadrilateral_coplanarity_check_1(scaling):
    with pytest.raises(RuntimeError) as error:
        # Unit square cell scaled down by 'scaling' and the first vertex
        # is distorted so that the vertices are clearly non coplanar
        mesh = Mesh(
            MPI.comm_world, CellType.quadrilateral,
            numpy.array(
                [[scaling, 0.5 * scaling, 0.6 * scaling], [0.0, scaling, 0.0],
                 [0.0, 0.0, scaling], [0.0, scaling, scaling]],
                dtype=numpy.float64),
            numpy.array([[0, 1, 2, 3]],
                        dtype=numpy.int32), [], cpp.mesh.GhostMode.none)

        mesh.create_connectivity_all()
        cpp.mesh.volume_entities(mesh, [0], mesh.topology.dim)

    assert "Not coplanar" in str(error.value)


# Test when |p0-p3| is ~ 1 but |p1-p2| is small
# The cell is degenerate when scale is below 1e-17, it is expected to fail the test.
@pytest.mark.parametrize('scaling', [1e0, 1e-5, 1e-10, 1e-15])
def test_volume_quadrilateral_coplanarity_check_2(scaling):
    with pytest.raises(RuntimeError) as error:
        # Unit square cell scaled down by 'scaling' and the first vertex
        # is distorted so that the vertices are clearly non coplanar
        mesh = Mesh(MPI.comm_world, CellType.quadrilateral,
                    numpy.array(
                        [[1.0, 0.5, 0.6], [0.0, scaling, 0.0],
                         [0.0, 0.0, scaling], [0.0, 1.0, 1.0]],
                        dtype=numpy.float64),
                    numpy.array([[0, 1, 2, 3]], dtype=numpy.int32), [],
                    cpp.mesh.GhostMode.none)
        mesh.create_connectivity_all()
        cpp.mesh.volume_entities(mesh, [0], mesh.topology.dim)

    assert "Not coplanar" in str(error.value)
