# Copyright (C) 2014 Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

from dolfin import (MPI, Expression, Function, FunctionSpace, UnitSquareMesh,
                    function)
from dolfin.io import HDF5File
from dolfin_utils.test.fixtures import tempdir
from dolfin_utils.test.skips import xfail_if_complex

assert (tempdir)


@xfail_if_complex
def test_save_and_read_function_timeseries(tempdir):
    filename = os.path.join(tempdir, "function.h5")

    mesh = UnitSquareMesh(MPI.comm_world, 10, 10)
    Q = FunctionSpace(mesh, ("CG", 3))
    F0 = Function(Q)
    F1 = Function(Q)

    t = 0.0

    @function.expression.numba_eval
    def expr_eval(values, x, cell_idx):
        values[:, 0] = t * x[:, 0]

    E = Expression(expr_eval)
    F0.interpolate(E)

    # Save to HDF5 File
    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "w")
    for t in range(10):
        F0.interpolate(E)
        hdf5_file.write(F0, "/function", t)
    hdf5_file.close()

    # Read back from file
    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "r")
    for t in range(10):
        F1.interpolate(E)
        vec_name = "/function/vector_%d" % t
        F0 = hdf5_file.read_function(Q, vec_name)
        # timestamp = hdf5_file.attributes(vec_name)["timestamp"]
        # assert timestamp == t
        F0.vector().axpy(-1.0, F1.vector())
        assert F0.vector().norm() < 1.0e-12
    hdf5_file.close()
