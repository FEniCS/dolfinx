#!/usr/bin/env py.test

"Unit tests for the MeshQuality class"

# Copyright (C) 2013 Garth N. Wells
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
# First added:  2013-10-07
# Last changed:

from __future__ import print_function
import pytest
import numpy
from dolfin import *
from dolfin_utils.test import skip_in_parallel


def test_radius_ratio_triangle():

    # Create mesh and compute rations
    mesh = UnitSquareMesh(12, 12)
    ratios = MeshQuality.radius_ratios(mesh)
    for c in cells(mesh):
        assert round(ratios[c] - 0.828427124746, 7) == 0


def test_radius_ratio_tetrahedron():

    # Create mesh and compute ratios
    mesh = UnitCubeMesh(14, 14, 14)
    ratios = MeshQuality.radius_ratios(mesh)
    for c in cells(mesh):
        assert round(ratios[c] - 0.717438935214, 7) == 0


def test_radius_ratio_triangle_min_max():

    # Create mesh, collpase and compute min ratio
    mesh = UnitSquareMesh(12, 12)

    rmin, rmax = MeshQuality.radius_ratio_min_max(mesh)
    assert rmax <= rmax

    x = mesh.coordinates()
    x[:, 0] *= 0.0
    rmin, rmax = MeshQuality.radius_ratio_min_max(mesh)
    assert round(rmin - 0.0, 7) == 0
    assert round(rmax - 0.0, 7) == 0


def test_radius_ratio_tetrahedron_min_max():

    # Create mesh, collpase and compute min ratio
    mesh = UnitCubeMesh(12, 12, 12)

    rmin, rmax = MeshQuality.radius_ratio_min_max(mesh)
    assert rmax <= rmax

    x = mesh.coordinates()
    x[:, 0] *= 0.0
    rmin, rmax = MeshQuality.radius_ratio_min_max(mesh)
    assert round(rmax - 0.0, 7) == 0
    assert round(rmax - 0.0, 7) == 0


def test_radius_ratio_matplotlib():
    # Create mesh, collpase and compute min ratio
    mesh = UnitCubeMesh(12, 12, 12)
    test = MeshQuality.radius_ratio_matplotlib_histogram(mesh, 5)
    print(test)


@skip_in_parallel
def test_radius_ratio_min_radius_ratio_max():
    mesh1d = UnitIntervalMesh(4)
    mesh1d.coordinates()[4] = mesh1d.coordinates()[3]

    # Create 2D mesh with one equilateral triangle
    mesh2d = UnitSquareMesh(1, 1, 'left')
    mesh2d.coordinates()[3] += 0.5*(sqrt(3.0)-1.0)

    # Create 3D mesh with regular tetrahedron and degenerate cells
    mesh3d = UnitCubeMesh(1, 1, 1)
    mesh3d.coordinates()[2][0] = 1.0
    mesh3d.coordinates()[7][1] = 0.0
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
    mesh = UnitCubeMesh(2, 2, 2)
    dang_min, dang_max = MeshQuality.dihedral_angles_min_max(mesh)
    assert round(dang_min*(180/numpy.pi) - 45.0) == 0
    assert round(dang_max*(180/numpy.pi) - 90.0) == 0


def test_dihedral_angles_matplotlib():
    # Create mesh, collpase and compute min ratio
    mesh = UnitCubeMesh(12, 12, 12)
    test = MeshQuality.dihedral_angles_matplotlib_histogram(mesh, 5)
    print(test)
