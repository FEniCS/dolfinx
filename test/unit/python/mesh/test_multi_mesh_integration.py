#!/usr/bin/env py.test

"""Unit tests for MultiMesh Integration"""

# Copyright (C) 2014 Anders Logg and August Johansson
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
# First added:  2014-03-04
# Last changed: 2015-11-30

from __future__ import print_function
import numpy
import pytest
from dolfin import *
from six.moves import xrange as range

from dolfin_utils.test import skip_in_parallel


def triangulation_to_mesh_2d(triangulation):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh, 2, 2)
    num_cells = len(triangulation) / 6
    num_vertices = len(triangulation) / 2
    editor.init_cells(num_cells)
    editor.init_vertices(num_vertices)
    for i in range(num_cells):
        editor.add_cell(i, 3*i, 3*i + 1, 3*i + 2)
    for i in range(num_vertices):
        editor.add_vertex(i, triangulation[2*i], triangulation[2*i + 1])
    editor.close()
    return mesh


@skip_in_parallel
def test_integrate():
    # Create two meshes of the unit square
    mesh_0 = UnitSquareMesh(10, 10)
    mesh_1 = UnitSquareMesh(11, 11)

    # Translate
    pt = Point(0.632350,0.278498)
    mesh_1.translate(pt)
    exactarea = 2-(1-pt[0])*(1-pt[1])

    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.add(mesh_1)
    multimesh.build()

    for part in range(0, multimesh.num_parts()):
        #print(part)
        covered = multimesh.covered_cells(part)
        uncut = multimesh.uncut_cells(part)
        cut = multimesh.cut_cells(part)
        qr = multimesh.quadrature_rule_cut_cells(part)
        #print("covered")
        #print(covered)
        #print("uncut")
        #print(uncut)
        #print("cut")
        #print(cut)
        #print("quadrature")
        #print(qr)

    # area = assemble(L_0) + assemble(L_1)
    # MMA = MultiMeshAssembler()
    # area = MMA.assemble(L_multi)
    # area = assemble(L_multi)
    # assert round(area - exactarea, 7) == 0


    # # Translate second mesh randomly
    # #dx = Point(numpy.random.rand(),numpy.random.rand())
    # dx = Point(0.278498, 0.546881)
    # mesh_1.translate(dx)

    # exactvolume = (1 - abs(dx[0]))*(1 - abs(dx[1]))

    # # Compute triangulation volume using the quadrature rules
    # volume = 0
    # for c0 in cells(mesh_0):
    #     for c1 in cells(mesh_1):
    #         triangulation = c0.triangulate_intersection(c1)
    #         if (triangulation.size>0):
    #             # compute_quadrature_rule(triangulation,
    #             #                         2,2,1)
    #             compute_quadrature_rule(c0,1)

    #             tmesh = triangulation_to_mesh_2d(triangulation)
    #             for t in cells(tmesh):
    #                 volume += t.volume()



    # errorstring = "translation=" + str(dx[0]) + str(" ") + str(dx[1])
    # assert round(volume - exactvolume, 7, errorstring)
