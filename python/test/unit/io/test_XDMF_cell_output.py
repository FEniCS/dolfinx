"""Unit test for XDMF output of DG0"""

# Copyright (C) 2014 Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from dolfin import *
from dolfin_utils.test import *
from dolfin.la import PETScVector

ghost_mode = set_parameters_fixture("ghost_mode", ["shared_vertex", "none"])

@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_xdmf_cell_scalar_ghost(cd_tempdir, ghost_mode):
    n = 8
    mesh = UnitSquareMesh(MPI.comm_world, n, n)

    print(mesh)

    Q = FunctionSpace(mesh, "DG", 0)
    F = Function(Q)
    E = Expression("x[0]", degree=1)
    F.interpolate(E)

    with XDMFFile(mesh.mpi_comm(), "dg0.xdmf") as xdmf:
        xdmf.write(F)

    # with HDF5File(mesh.mpi_comm(), "dg0.h5", "r") as hdf:
    #     vec = PETScVector(mesh.mpi_comm())
    #     hdf.read(vec, "/VisualisationVector/0", False)

    # area = MPI.sum(mesh.mpi_comm(), sum(vec.get_local()))
    # assert abs(n*n - area) < 1e-9
