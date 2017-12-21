# -*- coding: utf-8 -*-
"""Shared skips for unit tests involving dolfin."""

# Copyright (C) 2014-2014 Aslak Wigdahl Bergersen
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

import pytest
from dolfin import *


# Skips with dependencies
skip_if_not_MPI = pytest.mark.skipif(not has_mpi(),
                                     reason="Skipping unit test(s) depending on MPI.")
skip_if_not_PETsc_or_not_slepc = pytest.mark.skipif(not has_linear_algebra_backend("PETSc") or not has_slepc(),
                                                    reason='Skipping unit test(s) depending on PETSc and slepc.')
skip_if_not_HDF5 = pytest.mark.skipif(not has_hdf5(),
                                      reason="Skipping unit test(s) depending on HDF5.")
skip_if_not_PETSc = pytest.mark.skipif(not has_linear_algebra_backend("PETSc"),
                                       reason="Skipping unit test(s) depending on PETSc.")
skip_if_not_petsc4py = pytest.mark.skipif(not has_petsc4py(),
                                          reason="Skipping unit test(s) depending on petsc4py.")
skip_if_not_SLEPc = pytest.mark.skipif(not has_slepc(),
                                       reason="Skipping unit test(s) depending on SLEPc.")

# Skips with respect to parallel or serial
xfail_in_parallel = pytest.mark.xfail(MPI.size(MPI.comm_world) > 1,
                                      reason="This test does not yet work in parallel.")
xfail_with_serial_hdf5_in_parallel = pytest.mark.xfail(MPI.size(MPI.comm_world) > 1 and not has_hdf5_parallel(),
                                                       reason="Serial HDF5 library cannot work in parallel.")
skip_in_parallel = pytest.mark.skipif(MPI.size(MPI.comm_world) > 1,
                                      reason="This test should only be run in serial.")
skip_with_serial_hdf5_in_parallel = pytest.mark.skipif(MPI.size(MPI.comm_world) > 1 and not has_hdf5_parallel(),
                                                       reason="Serial HDF5 library cannot work in parallel.")
skip_in_serial = pytest.mark.skipif(MPI.size(MPI.comm_world) <= 1,
                                    reason="This test should only be run in parallel.")

# Skips with respect to linear algebra index type
skip_64bit_int = pytest.mark.skipif(cpp.common.sizeof_la_index() == 8,
                                    reason="This test does not work with 64-bit linear algebra indices.")

# Skips with respect to build type
skip_in_debug = pytest.mark.skipif(has_debug(),
                                   reason="This test does not work in debug mode.")
skip_in_release = pytest.mark.skipif(not has_debug(),
                                     reason="This test does not work in release mode.")
