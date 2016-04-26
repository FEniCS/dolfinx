#!/usr/bin/env py.test

"Unit tests for ghosted meshes"

# Copyright (C) 2016 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
import pytest
import numpy
from dolfin import *
import os

from dolfin_utils.test import fixture, skip_in_parallel, xfail_in_parallel, cd_tempdir, pushpop_parameters


def xtest_ghost_vertex_1d(pushpop_parameters):
    parameters["ghost_mode"] = "shared_vertex"
    mesh = UnitIntervalMesh(20)
    #print("Test: {}".format(MPI.sum(mesh.mpi_comm(), mesh.num_cells())))


def xtest_ghost_facet_1d(pushpop_parameters):
    parameters["ghost_mode"] = "shared_facet"
    mesh = UnitIntervalMesh(20)


def test_ghost_vertex_2d(pushpop_parameters):
    parameters["reorder_cells_gps"] = True
    #parameters["reorder_vertices_gps"] = True
    parameters["ghost_mode"] = "shared_vertex"
    mesh = UnitSquareMesh(8, 8)


def xtest_ghost_facet_2d(pushpop_parameters):
    parameters["ghost_mode"] = "shared_facet"
    mesh = UnitSquareMesh(5, 7)


def xtest_ghost_vertex_3d(pushpop_parameters):
    parameters["ghost_mode"] = "shared_vertex"
    mesh = UnitCubeMesh(5, 7, 3)


def xtest_ghost_facet_2d(pushpop_parameters):
    parameters["ghost_mode"] = "shared_facet"
    mesh = UnitCubeMesh(5, 7, 3)
