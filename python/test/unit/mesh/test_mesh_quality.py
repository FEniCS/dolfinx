# Copyright (C) 2013 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import numpy
from dolfin import *
from dolfin_utils.test import skip_in_parallel


def test_radius_ratio_triangle():

    # Create mesh and compute rations
    mesh = UnitSquareMesh(MPI.comm_world, 12, 12)
    ratios = MeshQuality.radius_ratios(mesh)
    for c in Cells(mesh):
        assert round(ratios[c] - 0.828427124746, 7) == 0


def test_radius_ratio_tetrahedron():

    # Create mesh and compute ratios
    mesh = UnitCubeMesh(MPI.comm_world, 14, 14, 14)
    ratios = MeshQuality.radius_ratios(mesh)
    for c in Cells(mesh):
        assert round(ratios[c] - 0.717438935214, 7) == 0


def test_radius_ratio_triangle_min_max():

    # Create mesh, collpase and compute min ratio
    mesh = UnitSquareMesh(MPI.comm_world, 12, 12)

    rmin, rmax = MeshQuality.radius_ratio_min_max(mesh)
    assert rmax <= rmax

    x = mesh.geometry().x()
    x[:, 0] *= 0.0
    rmin, rmax = MeshQuality.radius_ratio_min_max(mesh)
    assert round(rmin - 0.0, 7) == 0
    assert round(rmax - 0.0, 7) == 0


def test_radius_ratio_tetrahedron_min_max():

    # Create mesh, collpase and compute min ratio
    mesh = UnitCubeMesh(MPI.comm_world, 12, 12, 12)

    rmin, rmax = MeshQuality.radius_ratio_min_max(mesh)
    assert rmax <= rmax

    x = mesh.geometry().x()
    x[:, 0] *= 0.0
    rmin, rmax = MeshQuality.radius_ratio_min_max(mesh)
    assert round(rmax - 0.0, 7) == 0
    assert round(rmax - 0.0, 7) == 0


def test_radius_ratio_matplotlib():
    # Create mesh, collpase and compute min ratio
    mesh = UnitCubeMesh(MPI.comm_world, 12, 12, 12)
    test = MeshQuality.radius_ratio_matplotlib_histogram(mesh, 5)
    print(test)


@skip_in_parallel
def test_radius_ratio_min_radius_ratio_max():
    mesh1d = UnitIntervalMesh(MPI.comm_self, 4)
    x = mesh1d.geometry().x()
    x[4] = mesh1d.geometry().x()[3]

    # Create 2D mesh with one equilateral triangle
    mesh2d = RectangleMesh.create(MPI.comm_world, [Point(0,0), Point(1,1)], [1, 1], CellType.Type.triangle, 'left')
    x = mesh2d.geometry().x()
    x[3] += 0.5*(sqrt(3.0)-1.0)

    # Create 3D mesh with regular tetrahedron and degenerate cells
    mesh3d = UnitCubeMesh(MPI.comm_self, 1, 1, 1)
    x = mesh3d.geometry().x()
    x[6][0] = 1.0
    x[3][1] = 0.0
    rmin, rmax = MeshQuality.radius_ratio_min_max(mesh1d)
    assert round(rmin - 0.0, 7) == 0
    assert round(rmax - 1.0, 7) == 0

    rmin, rmax = MeshQuality.radius_ratio_min_max(mesh2d)
    assert round(rmin - 2.0*sqrt(2.0)/(2.0+sqrt(2.0)), 7) == 0
    assert round(rmax - 1.0, 7) == 0

    rmin, rmax = MeshQuality.radius_ratio_min_max(mesh3d)
    assert round(rmin - 0.0, 7) == 0
    assert round(rmax - 1.0, 7) == 0


def test_dihedral_angles_min_max():
    # Create 3D mesh with regular tetrahedron
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    dang_min, dang_max = MeshQuality.dihedral_angles_min_max(mesh)
    assert round(dang_min*(180/numpy.pi) - 45.0) == 0
    assert round(dang_max*(180/numpy.pi) - 90.0) == 0


def test_dihedral_angles_matplotlib():
    # Create mesh, collpase and compute min ratio
    mesh = UnitCubeMesh(MPI.comm_world, 12, 12, 12)
    test = MeshQuality.dihedral_angles_matplotlib_histogram(mesh, 5)
    print(test)
