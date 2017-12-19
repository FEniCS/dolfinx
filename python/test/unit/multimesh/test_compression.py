"""Unit tests for quadrature compression"""

# Copyright (C) 2017 August Johansson
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
##
# First added:  2016-06-28
# Last changed: 2017-05-28

import pytest

from dolfin import *
from dolfin_utils.test import skip_in_parallel
from numpy import random, sort, any

def build_multimesh_2d(compress_volume, compress_interface):

    # Create multimesh
    multimesh = MultiMesh()
    multimesh.parameters["compress_volume_quadrature"] = compress_volume
    multimesh.parameters["compress_interface_quadrature"] = compress_interface

    # Add background mesh
    N_x = 2
    mesh = UnitSquareMesh(N_x, N_x)
    multimesh.add(mesh)

    # Set seed
    random.seed(1)

    # Add num_parts-1 random sized and rotated rectangular meshes
    num_parts = 2
    while (multimesh.num_parts() < num_parts):

        x0, x1 = sort(random.rand(2))
        y0, y1 = sort(random.rand(2))
        if abs(x1 - x0) < DOLFIN_EPS:
            x1 += DOLFIN_EPS
        if abs(y1 - y0) < DOLFIN_EPS:
            y1 += DOLFIN_EPS

        N_x_part = int(max(abs(x1-x0)*N_x, 1))
        N_y_part = int(max(abs(y1-y0)*N_x, 1))
        mesh = RectangleMesh(Point(x0, y0), Point(x1, y1),
                             N_x_part, N_y_part)

        # Rotate
        phi = random.rand()*180
        mesh.rotate(phi)
        coords = mesh.coordinates()
        is_interior = not any(coords < 0) and not any(coords > 1.)

        if is_interior:
            multimesh.add(mesh)

    multimesh.build()
    return multimesh

def volume_area(multimesh):
    volume = multimesh.compute_volume()
    area = multimesh.compute_area()
    return volume, area

@skip_in_parallel
@pytest.mark.skip
def test_compression_2d():
    # Reference volume and area
    multimesh = build_multimesh_2d(False, False)
    volume, area = volume_area(multimesh)

    # Volume compression
    multimesh_v = build_multimesh_2d(True, False)
    volume_v, area_v = volume_area(multimesh)

    # Interface compression
    multimesh_i = build_multimesh_2d(False, True)
    volume_i, area_i = volume_area(multimesh_i)

    # Tolerances
    volume_tol = 1e-10
    area_tol = 1e-10

    # Assert that the compressed volume / area is close
    assert abs(volume - volume_v) < volume_tol
    assert abs(area - area_i) < area_tol

    # Assert that the compressed volume / area does not influence the area / volume
    assert abs(volume - volume_i) < DOLFIN_EPS
    assert abs(area - area_v) < DOLFIN_EPS
