#!/usr/bin/env py.test

"""Unit test for HarmonicSmoothing and ALE"""

# Copyright (C) 2013 Jan Blechta
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

from __future__ import print_function
import pytest
from dolfin import UnitSquareMesh, BoundaryMesh, Expression, \
                   CellFunction, SubMesh, Constant, MPI, MeshQuality,\
                   mpi_comm_world, ALE
from dolfin_utils.test import skip_in_parallel

def test_HarmonicSmoothing():

    # Create some mesh and its boundary
    mesh = UnitSquareMesh(10, 10)
    boundary = BoundaryMesh(mesh, 'exterior')

    # Move boundary
    disp = Expression(("0.3*x[0]*x[1]", "0.5*(1.0-x[1])"), degree=2)
    ALE.move(boundary, disp)

    # Move mesh according to given boundary
    ALE.move(mesh, boundary)

    # Check that new boundary topology corresponds to given one
    boundary_new = BoundaryMesh(mesh, 'exterior')
    assert boundary.topology().hash() == boundary_new.topology().hash()

    # Check that coordinates are almost equal
    err = sum(sum(abs(boundary.coordinates() \
                    - boundary_new.coordinates()))) / mesh.num_vertices()
    print("Current CG solver produced error in boundary coordinates", err)
    assert round(err - 0.0, 5) == 0

    # Check mesh quality
    magic_number = 0.35
    rmin = MeshQuality.radius_ratio_min_max(mesh)[0]
    assert rmin > magic_number

@skip_in_parallel
def test_ale():

    # Create some mesh
    mesh = UnitSquareMesh(4, 5)

    # Make some cell function
    # FIXME: Initialization by array indexing is probably
    #        not a good way for parallel test
    cellfunc = CellFunction('size_t', mesh)
    cellfunc.array()[0:4] = 0
    cellfunc.array()[4:]  = 1

    # Create submeshes - this does not work in parallel
    submesh0 = SubMesh(mesh, cellfunc, 0)
    submesh1 = SubMesh(mesh, cellfunc, 1)

    # Move submesh0
    disp = Constant(("0.1", "-0.1"))
    ALE.move(submesh0, disp)

    # Move and smooth submesh1 accordignly
    ALE.move(submesh1, submesh0)

    # Move mesh accordingly
    parent_vertex_indices_0 = \
        submesh0.data().array('parent_vertex_indices', 0)
    parent_vertex_indices_1 = \
        submesh1.data().array('parent_vertex_indices', 0)
    mesh.coordinates()[parent_vertex_indices_0[:]] = \
        submesh0.coordinates()[:]
    mesh.coordinates()[parent_vertex_indices_1[:]] = \
        submesh1.coordinates()[:]

    # If test passes here then it is probably working
    # Check for cell quality for sure
    magic_number = 0.28
    rmin = MeshQuality.radius_ratio_min_max(mesh)[0]
    assert rmin > magic_number
