#!/usr/bin/env py.test

"""Unit tests for multimesh volume computation"""

# Copyright (C) 2016 Anders Logg
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
# Modified by August Johansson 2016
#
# First added:  2016-11-02
# Last changed: 2016-11-10

from __future__ import print_function
import pytest

from dolfin import *
from dolfin_utils.test import skip_in_parallel

def compute_area_using_quadrature(multimesh):
    total_area = 0
    for part in range(multimesh.num_parts()):
        part_area = 0

        for cut_cell in multimesh.collision_map_cut_cells(part):
            local_qr = multimesh.quadrature_rule_interface_cut_cell(part, cut_cell)
            for weight in local_qr[1]:
                part_area += weight

        total_area += part_area
    return total_area


def create_multimesh_with_meshes_on_diagonal(width, offset, Nx):

    # Mesh width (must be less than 1)
    assert width < 1

    # Mesh placement (must be less than the width)
    assert offset < width

    # Background mesh
    mesh_0 = UnitSquareMesh(Nx, Nx)

    # Create multimesh
    multimesh = MultiMesh()
    multimesh.add(mesh_0)

    # Now we have num_parts = 1
    num_parts = multimesh.num_parts()

    while num_parts*offset + width < 1:
        a = num_parts*offset
        b = a + width
        mesh_top = RectangleMesh(Point(a,a), Point(b,b), Nx, Nx)
        multimesh.add(mesh_top)
        num_parts = multimesh.num_parts()

    multimesh.build()

    area = compute_area_using_quadrature(multimesh)
    exact_area = 0 if multimesh.num_parts() == 1 else 4*width + (multimesh.num_parts()-2)*(2*width + 2*offset)
    error = abs(area - exact_area)
    relative_error = error / exact_area
    tol = max(DOLFIN_EPS_LARGE, multimesh.num_parts()*multimesh.part(0).num_cells()*DOLFIN_EPS)
    print("width = {}, offset = {}, Nx = {}, num_parts = {}".format(width, offset, Nx, multimesh.num_parts()))
    print("error", error)
    print("relative error", relative_error)
    print("tol", tol)
    return relative_error < tol

def area_equal(multimesh, area, exact_area):
    tol = max(DOLFIN_EPS_LARGE, multimesh.num_parts()*multimesh.part(0).num_cells()*DOLFIN_EPS)
    error = abs(area - exact_area) / exact_area
    return error < tol

@skip_in_parallel
def test_meshes_on_diagonal():
    "Place meshes on the diagonal inside a background mesh and check the interface area"

    width = 0.3
    offset = 0.1111
    for Nx in range(1, 50):
        assert(create_multimesh_with_meshes_on_diagonal(width, offset, Nx))

    # width = 0.6
    # offset = 0.1
    # for Nx in range(1, 50):
    #     assert(create_multimesh_with_meshes_on_diagonal(width, offset, Nx))

    width = 1/DOLFIN_PI #0.18888
    offset = 0.04 #1e-10
    for Nx in range(1, 50):
        assert(create_multimesh_with_meshes_on_diagonal(width, offset, Nx))

# @skip_in_parallel
# def test_meshes_with_boundary_edge_overlap_2D():
#     # start with boundary of mesh 1 overlapping edges of mesg 0
#     mesh0 = UnitSquareMesh(4,4)
#     mesh1 = UnitSquareMesh(1,1)

#     mesh1_coords = mesh1.coordinates()
#     mesh1_coords *= 0.5
#     mesh1.translate(Point(0.25, 0.25))

#     multimesh = MultiMesh()
#     multimesh.add(mesh0)
#     multimesh.add(mesh1)
#     multimesh.build()

#     exact_area = 2.0

#     area = compute_area_using_quadrature(multimesh)
#     assert  abs(area - exact_area) < DOLFIN_EPS_LARGE

#     # next translate mesh 1 such that only the horizontal part of the boundary overlaps
#     mesh1.translate(Point(0.1, 0.0))
#     multimesh.build()
#     area = compute_area_using_quadrature(multimesh)
#     assert  abs(area - exact_area) < DOLFIN_EPS_LARGE

#     # next translate mesh 1 such that no boundaries overlap with edges
#     mesh1.translate(Point(0.0, 0.1))
#     multimesh.build()
#     area = compute_area_using_quadrature(multimesh)
#     assert  abs(area - exact_area) < DOLFIN_EPS_LARGE



# # FIXME: Temporary testing
# if __name__ == "__main__":
#     test_meshes_on_diagonal()
#     test_meshes_with_boundary_edge_overlap_2D()
