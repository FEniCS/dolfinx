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
# First added:  2016-05-03
# Last changed: 2016-07-05

from __future__ import print_function
import pytest

from dolfin import *
from dolfin_utils.test import skip_in_parallel

def compute_volume(multimesh):
    # Create function space
    V = MultiMeshFunctionSpace(multimesh, "DG", 0)

    # Create and evaluate volume functional
    v = TestFunction(V)
    M = v*dX
    return sum(assemble(M).array())

    # M = Constant(1.0)*dX(multimesh)
    # return assemble(M)

def compute_volume_using_quadrature(multimesh):
    total_volume = 0
    for part in range(multimesh.num_parts()):
        mesh = multimesh.part(part)
        part_volume = 0

        # Volume of uncut cells
        for uncut_cell in multimesh.uncut_cells(part):
            cell = Cell(mesh, uncut_cell)
            part_volume += cell.volume()

        # Volume of cut cells
        for cut_cell in multimesh.cut_cells(part):
            cut_cell_qr = multimesh.quadrature_rule_cut_cell(part, cut_cell)
            for weight in cut_cell_qr[1]:
                part_volume += weight

        total_volume += part_volume
    return total_volume


@skip_in_parallel
def test_volume_2d():
    "Integrate volume of union of 2D meshes"

    # Number of meshes
    num_meshes = 2

    # Create background mesh so we can easily compute volume...
    mesh_0 = UnitSquareMesh(1, 1)
    mesh_0.scale(10.0)
    mesh_0.translate(Point(-5, -5))
    exact_volume = 100.0

    # Create meshes with centres distributed around the unit circle.
    # Meshes are scaled by a factor 2 so that they overlap.
    meshes = []
    for i in range(num_meshes):
        mesh = UnitSquareMesh(1, 1)
        angle = 2.*pi*float(i) / float(num_meshes)
        mesh.translate(Point(-0.5, -0.5))
        mesh.scale(2.0)
        mesh.rotate(180.0*angle / pi)
        mesh.translate(Point(cos(angle), sin(angle)))
        meshes.append(mesh)

    # Save meshes to file so we can examine them
    File('output/background_mesh.pvd') << mesh_0
    vtkfile = File('output/meshes.pvd')
    for mesh in meshes:
        vtkfile << mesh

    # Create multimesh
    multimesh = MultiMesh()
    for mesh in [mesh_0] + meshes:
        multimesh.add(mesh)
    multimesh.build()

    # Create function space
    V = MultiMeshFunctionSpace(multimesh, "DG", 0)

    # Compute approximate volume
    approximative_volume = compute_volume(multimesh)

    print("approximative volume ", approximative_volume)
    print("exact volume ", exact_volume)

    assert approximative_volume == exact_volume

@skip_in_parallel
def test_volume_six_meshes():
    "Integrate volume of six 2D meshes"

    # Number of elements
    Nx = 8
    h = 1. / Nx

    # Background mesh
    mesh_0 = UnitSquareMesh(Nx, Nx)

    # 5 meshes plus background mesh
    num_meshes = 5

    # List of points for generating the meshes on top
    points = [[ Point(0.747427, 0.186781), Point(0.849659, 0.417130) ],
              [ Point(0.152716, 0.471681), Point(0.455943, 0.741585) ],
              [ Point(0.464473, 0.251876), Point(0.585051, 0.533569) ],
              [ Point(0.230112, 0.511897), Point(0.646974, 0.892193) ],
              [ Point(0.080362, 0.422675), Point(0.580151, 0.454286) ]]

    angles = [ 88.339755, 94.547259, 144.366564, 172.579922, 95.439692 ]

    # Create multimesh
    multimesh = MultiMesh()
    multimesh.add(mesh_0)

    # Add the 5 background meshes
    for i in range(num_meshes):
        nx = max(int(round(abs(points[i][0].x()-points[i][1].x()) / h)), 1)
        ny = max(int(round(abs(points[i][0].y()-points[i][1].y()) / h)), 1)
        mesh = RectangleMesh(points[i][0], points[i][1], nx, ny)
        mesh.rotate(angles[i])
        multimesh.add(mesh)
    multimesh.build()

    # Save meshes to file
    vtkfile = File('output/meshes.pvd')
    for i in range(multimesh.num_parts()):
        vtkfile << multimesh.part(i)

    exact_volume = 1
    approximate_volume = compute_volume(multimesh)
    qr_volume = compute_volume_using_quadrature(multimesh)

    print("exact volume ", exact_volume)
    print("qr volume ", qr_volume)
    print("error %1.16e" % (exact_volume - qr_volume))

    assert abs(exact_volume - qr_volume) < DOLFIN_EPS_LARGE

# FIXME: Temporary testing
if __name__ == "__main__":
    test_volume_2d()
    test_volume_six_meshes()
