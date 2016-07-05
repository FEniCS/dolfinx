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

from math import pi, sin, cos

@skip_in_parallel
def compute_volume(multimesh):
    # Create function space
    V = MultiMeshFunctionSpace(multimesh, "DG", 0)

    # Create and evaluate volume functional
    v = TestFunction(V)
    M = v*dX
    return sum(assemble(M).array())

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
def test_volume_7_meshes():
    "Integrate volume of 7 2D meshes"

    # Number of elements
    Nx = 8
    h = 1. / Nx

    # Background mesh
    mesh_0 = UnitSquareMesh(Nx, Nx)

    # 6 meshes plus background mesh
    num_meshes = 6

    # List of points for generating the 6 meshes on top
    points = [ 0.747427, 0.186781, 0.849659, 0.417130,
               0.152716, 0.471681, 0.455943, 0.741585,
               0.464473, 0.251876, 0.585051, 0.533569,
               0.230112, 0.511897, 0.646974, 0.892193,
               0.080362, 0.422675, 0.580151, 0.454286,
               0.054755, 0.534186, 0.444096, 0.743028 ]
    angles = [ 88.339755, 94.547259, 144.366564, 172.579922, 95.439692, 106.697958 ]

    # Create multimesh
    multimesh = MultiMesh()
    multimesh.add(mesh_0)

    # Add the 6 background meshes
    for i in range(num_meshes):
        nx = max(int(round(abs(points[4*i]-points[4*i+2]) / h)), 1)
        ny = max(int(round(abs(points[4*i+1]-points[4*i+3]) / h)), 1)
        mesh = RectangleMesh(Point(points[4*i], points[4*i+1]),
                             Point(points[4*i+2], points[4*i+3]),
                             nx, ny)
        mesh.rotate(angles[i])
        multimesh.add(mesh)
    multimesh.build()

    # Save meshes to file
    vtkfile = File('output/meshes.pvd')
    for i in range(multimesh.num_parts()):
        vtkfile << multimesh.part(i)

    exact_volume = 1
    approximate_volume = compute_volume(multimesh)

    print("approximate volume ", approximate_volume)
    print("exact volume ", exact_volume)
    print("error %1.16f" % (exact_volume - approximate_volume))

    assert exact_volume == approximate_volume

# FIXME: Temporary testing
if __name__ == "__main__":
    #test_volume_2d()
    test_volume_7_meshes()
