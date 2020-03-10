# Copyright (C) 2012 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

from dolfinx import MPI
from dolfinx.io import HDF5File
from dolfinx_utils.test.fixtures import tempdir

assert (tempdir)


def test_parallel(tempdir):
    filename = os.path.join(tempdir, "y.h5")
    hdf5 = HDF5File(MPI.comm_world, filename, "w")
    assert (hdf5)


def test_mpi_atomicity(tempdir):
    comm_world = MPI.comm_world
    if MPI.size(comm_world) > 1:
        filename = os.path.join(tempdir, "mpiatomic.h5")
        with HDF5File(MPI.comm_world, filename, "w") as f:
            assert f.get_mpi_atomicity() is False
            f.set_mpi_atomicity(True)
            assert f.get_mpi_atomicity() is True
