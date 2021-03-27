# Copyright (C) 2021 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# import os

import pytest
from dolfinx.cpp.io import has_adios2
from dolfinx.generation import UnitCubeMesh
# from dolfinx import (Function, FunctionSpace, TensorFunctionSpace,
#                      UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh,
#                      VectorFunctionSpace)
# from dolfinx.cpp.mesh import CellType
# from dolfinx.io import VTKFile
# from dolfinx_utils.test.fixtures import tempdir
# from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI

# assert (tempdir)


@pytest.mark.skipif(not has_adios2, reason="Requires ADIOS2.")
def test_save_mesh():
    from dolfinx.cpp.io import ADIOS2File
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    f = ADIOS2File(mesh.mpi_comm(), "mesh.bp", "w")
    f.write_mesh(mesh)
