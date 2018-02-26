# Copyright (C) 2014 Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import os
from dolfin import *
from dolfin_utils.test import skip_if_not_HDF5, fixture, tempdir, \
    xfail_with_serial_hdf5_in_parallel

@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_save_and_read_function_timeseries(tempdir):
    filename = os.path.join(tempdir, "function.h5")

    mesh = UnitSquareMesh(MPI.comm_world, 10, 10)
    Q = FunctionSpace(mesh, "CG", 3)
    F0 = Function(Q)
    F1 = Function(Q)
    E = Expression("t*x[0]", t = 0.0, degree=1)
    F0.interpolate(E)

    # Save to HDF5 File
    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "w")
    for t in range(10):
        E.t = t
        F0.interpolate(E)
        hdf5_file.write(F0, "/function", t)
    hdf5_file.close()

    #Read back from file
    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "r")
    for t in range(10):
        E.t = t
        F1.interpolate(E)
        vec_name = "/function/vector_%d"%t
        hdf5_file.read(F0, vec_name)
        #timestamp = hdf5_file.attributes(vec_name)["timestamp"]
        #assert timestamp == t
        result = F0.vector() - F1.vector()
        assert len(result.get_local().nonzero()[0]) == 0
    hdf5_file.close()
