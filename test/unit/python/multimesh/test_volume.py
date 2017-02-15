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
# Modified by Simon Funke 2017
#
# First added:  2016-05-03
# Last changed: 2017-02-15

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
    v1 = sum(assemble(M).array())

    # Alternative volume computation
    dXmm = dx(domain=multimesh) + dC(domain=multimesh)
    M = Constant(1.0)*dXmm
    v2 = assemble_multimesh(M)

    # FIXME: We should be able to tighten the tolerance here
    assert abs(v1 - v2) < 100*DOLFIN_EPS_LARGE  
    return v1

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
            cut_cell_qr = multimesh.quadrature_rule_cut_cells(part, cut_cell)
            if cut_cell_qr:
                part_volume += sum(cut_cell_qr[1])

        # Volume of covered cells
        for covered_cell in multimesh.covered_cells(part):
            cell = Cell(mesh, covered_cell)
            part_volume += cell.volume()

        print("part volume", part, part_volume)
        total_volume += part_volume
    return total_volume


@skip_in_parallel
def test_volume_2d():
    "Integrate volume of union of 2D meshes"

    # Number of meshes on top of background mesh
    num_meshes = 8

    # Create background mesh so we can easily compute volume
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
        print("i, angle", i, angle)
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

    # Compute approximate volume
    approximative_volume = compute_volume(multimesh)
    qr_volume = compute_volume_using_quadrature(multimesh)

    print("exact volume ", exact_volume)
    print("approximative volume ", approximative_volume)
    print("relative approximate volume error %1.16e" % ((exact_volume - approximative_volume) / exact_volume))
    print("qr volume ", qr_volume)
    print("relative qr volume error %1.16e" % ((exact_volume - qr_volume) / exact_volume))

    assert (abs(exact_volume - qr_volume) / exact_volume) < DOLFIN_EPS_LARGE
    assert abs(exact_volume - approximative_volume) / exact_volume < DOLFIN_EPS

# @skip_in_parallel
# def test_volume_six_meshes():
#     "Integrate volume of six 2D meshes"

#     # Number of elements
#     Nx = 8
#     h = 1. / Nx

#     # Background mesh
#     mesh_0 = UnitSquareMesh(Nx, Nx)

#     # 5 meshes plus background mesh
#     num_meshes = 5

#     # List of points for generating the meshes on top
#     points = [[ Point(0.747427, 0.186781), Point(0.849659, 0.417130) ],
#               [ Point(0.152716, 0.471681), Point(0.455943, 0.741585) ],
#               [ Point(0.464473, 0.251876), Point(0.585051, 0.533569) ],
#               [ Point(0.230112, 0.511897), Point(0.646974, 0.892193) ],
#               [ Point(0.080362, 0.422675), Point(0.580151, 0.454286) ]]

#     angles = [ 88.339755, 94.547259, 144.366564, 172.579922, 95.439692 ]

#     # Create multimesh
#     multimesh = MultiMesh()
#     multimesh.add(mesh_0)

#     # Add the 5 background meshes
#     for i in range(num_meshes):
#         nx = max(int(round(abs(points[i][0].x()-points[i][1].x()) / h)), 1)
#         ny = max(int(round(abs(points[i][0].y()-points[i][1].y()) / h)), 1)
#         mesh = RectangleMesh(points[i][0], points[i][1], nx, ny)
#         mesh.rotate(angles[i])
#         multimesh.add(mesh)
#     multimesh.build()

#     # Save meshes to file
#     vtkfile = File('output/test_six_meshes.pvd')
#     for i in range(multimesh.num_parts()):
#         vtkfile << multimesh.part(i)

#     exact_volume = 1
#     approximate_volume = compute_volume(multimesh)
#     qr_volume = compute_volume_using_quadrature(multimesh)

#     print("exact volume ", exact_volume)
#     print("approximative volume ", approximate_volume)
#     print("approximate volume error %1.16e" % (exact_volume - approximate_volume))
#     print("qr volume ", qr_volume)
#     print("qr volume error %1.16e" % (exact_volume - qr_volume))

#     assert abs(exact_volume - qr_volume) < 7e-15
#     assert abs(exact_volume - approximate_volume) < DOLFIN_EPS

# @skip_in_parallel
# def test_main2_volume():
#     "Test with four meshes that previously failed"

#     # Create multimesh
#     multimesh = MultiMesh()

#     # Mesh size
#     h = 0.25
#     Nx = int(round(1 / h))

#     # Background mesh
#     mesh_0 = UnitSquareMesh(Nx, Nx)
#     multimesh.add(mesh_0)

#     # Mesh 1
#     x0 = 0.35404867974764142602
#     y0 = 0.16597416632155614913
#     x1 = 0.63997881656511634851
#     y1 = 0.68786139026650294781
#     mesh_1 = RectangleMesh(Point(x0, y0), Point(x1, y1),
#                            max(int(round((x1-x0)/h)), 1), max(int(round((y1-y0)/h)), 1))
#     mesh_1.rotate(39.609407484349517858)
#     multimesh.add(mesh_1)

#     # Mesh 2
#     x0 = 0.33033712968711609337
#     y0 = 0.22896817104377231722
#     x1 = 0.82920109332967595339
#     y1 = 0.89337241458397931293
#     mesh_2 = RectangleMesh(Point(x0, y0), Point(x1, y1),
#                            max(int(round((x1-x0)/h)), 1), max(int(round((y1-y0)/h)), 1))
#     mesh_2.rotate(31.532416069662392744)
#     multimesh.add(mesh_2)

#     # Mesh 3
#     x0 = 0.28105941241656401397
#     y0 = 0.30745787374091237965
#     x1 = 0.61959648394007071914
#     y1 = 0.78600209801737319637
#     mesh_3 = RectangleMesh(Point(x0, y0), Point(x1, y1),
#                            max(int(round((x1-x0)/h)), 1), max(int(round((y1-y0)/h)), 1))
#     mesh_3.rotate(40.233022128340330426)
#     multimesh.add(mesh_3)

#     multimesh.build()

#     exact_volume = 1
#     approximate_volume = compute_volume(multimesh)
#     qr_volume = compute_volume_using_quadrature(multimesh)

#     print("exact volume ", exact_volume)
#     print("approximative volume ", approximate_volume)
#     print("approximate volume error %1.16e" % (exact_volume - approximate_volume))
#     print("qr volume ", qr_volume)
#     print("qr volume error %1.16e" % (exact_volume - qr_volume))

#     assert abs(exact_volume - qr_volume) < 7e-16
#     assert abs(exact_volume - approximate_volume) < DOLFIN_EPS
