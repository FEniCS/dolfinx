# Copyright (C) 2016 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import numpy
from dolfin import *
import os

from dolfin_utils.test import (fixture, skip_in_parallel,
                               xfail_in_parallel, cd_tempdir,
                               pushpop_parameters)
from dolfin.parameter import parameters

# See https://bitbucket.org/fenics-project/dolfin/issues/579


def xtest_ghost_vertex_1d(pushpop_parameters):
    parameters["ghost_mode"] = "shared_vertex"
    mesh = UnitIntervalMesh(MPI.comm_world, 20)
    #print("Test: {}".format(MPI.sum(mesh.mpi_comm(), mesh.num_cells())))


def xtest_ghost_facet_1d(pushpop_parameters):
    parameters["ghost_mode"] = "shared_facet"
    mesh = UnitIntervalMesh(MPI.comm_world, 20)


@pytest.mark.xfail
def test_ghost_2d(pushpop_parameters):
    modes = ["shared_vertex", "shared_facet"]
    for mode in modes:
        parameters["ghost_mode"] = mode
        N = 8
        num_cells = 128

        mesh = UnitSquareMesh(MPI.comm_world, N, N)
        if MPI.size(mesh.mpi_comm()) > 1:
            assert MPI.sum(mesh.mpi_comm(), mesh.num_cells()) > num_cells

        parameters["reorder_cells_gps"] = True
        parameters["reorder_vertices_gps"] = False
        mesh = UnitSquareMesh(MPI.comm_world, N, N)
        if MPI.size(mesh.mpi_comm()) > 1:
            assert MPI.sum(mesh.mpi_comm(), mesh.num_cells()) > num_cells

        parameters["reorder_cells_gps"] = True
        parameters["reorder_vertices_gps"] = True
        mesh = UnitSquareMesh(MPI.comm_world, N, N)
        if MPI.size(mesh.mpi_comm()) > 1:
            assert MPI.sum(mesh.mpi_comm(), mesh.num_cells()) > num_cells

        parameters["reorder_cells_gps"] = False
        parameters["reorder_vertices_gps"] = True
        mesh = UnitSquareMesh(MPI.comm_world, N, N)
        if MPI.size(mesh.mpi_comm()) > 1:
            assert MPI.sum(mesh.mpi_comm(), mesh.num_cells()) > num_cells


@pytest.mark.xfail
def test_ghost_3d(pushpop_parameters):
    modes = ["shared_vertex", "shared_facet"]
    for mode in modes:
        parameters["ghost_mode"] = mode
        N = 2
        num_cells = 48

        mesh = UnitCubeMesh(MPI.comm_world, N, N, N)
        if MPI.size(mesh.mpi_comm()) > 1:
            assert MPI.sum(mesh.mpi_comm(), mesh.num_cells()) > num_cells

        parameters["reorder_cells_gps"] = True
        parameters["reorder_vertices_gps"] = False
        mesh = UnitCubeMesh(MPI.comm_world, N, N, N)
        if MPI.size(mesh.mpi_comm()) > 1:
            assert MPI.sum(mesh.mpi_comm(), mesh.num_cells()) > num_cells

        parameters["reorder_cells_gps"] = True
        parameters["reorder_vertices_gps"] = True
        mesh = UnitCubeMesh(MPI.comm_world, N, N, N)
        if MPI.size(mesh.mpi_comm()) > 1:
            assert MPI.sum(mesh.mpi_comm(), mesh.num_cells()) > num_cells

        parameters["reorder_cells_gps"] = False
        parameters["reorder_vertices_gps"] = True
        mesh = UnitCubeMesh(MPI.comm_world, N, N, N)
        if MPI.size(mesh.mpi_comm()) > 1:
            assert MPI.sum(mesh.mpi_comm(), mesh.num_cells()) > num_cells


@pytest.mark.xfail
@pytest.mark.parametrize('gmode', ['shared_vertex', 'shared_facet', 'none'])
def test_ghost_connectivities(gmode, pushpop_parameters):
    parameters['ghost_mode'] = gmode

    # Ghosted mesh
    meshG = UnitSquareMesh(MPI.comm_world, 4, 4)
    meshG.init(1, 2)

    # Reference mesh, not ghosted, not parallel
    meshR = UnitSquareMesh(MPI.comm_self, 4, 4)
    meshR.init(1, 2)

    # Create reference mapping from facet midpoint to cell midpoint
    reference = {}
    for facet in Facets(meshR):
        fidx = facet.index()
        facet_mp = tuple(facet.midpoint()[:])
        reference[facet_mp] = []
        for cidx in meshR.topology()(1, 2)(fidx):
            cell = Cell(meshR, cidx)
            cell_mp = tuple(cell.midpoint()[:])
            reference[facet_mp].append(cell_mp)

    # Loop through ghosted mesh and check connectivities
    allowable_cell_indices = [cell.index() for cell in Cells(meshG, 'all')]
    for facet in Facets(meshG, 'regular'):
        fidx = facet.index()
        facet_mp = tuple(facet.midpoint()[:])
        assert facet_mp in reference

        for cidx in meshG.topology()(1, 2)(fidx):
            assert cidx in allowable_cell_indices
            cell = Cell(meshG, cidx)
            cell_mp = tuple(cell.midpoint()[:])
            assert cell_mp in reference[facet_mp]
