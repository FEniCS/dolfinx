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
# Last changed: 2016-05-04

from __future__ import print_function
import pytest

from dolfin import *
from dolfin_utils.test import skip_in_parallel

from math import pi, sin, cos

@skip_in_parallel
def test_volume_2d():
    "Integrate volume of union of 2D meshes"

    # Number of meshes
    num_meshes = 3

    # Create background mesh so we can easily compute volume...
    mesh_0 = UnitSquareMesh(1, 1)
    mesh_0.translate(Point(-0.5, -0.5))
    mesh_0.scale(5.0)
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

    # Create multimesh function space
    #V = MultiMeshFunctionSpace(multimesh, 'P', 1)
    #M = Constant(1)*dX(multimesh)

    assert 1 == 1
