"""Shared skips for unit tests involving dolfin."""

# Copyright (C) 2014-2014 Aslak Wigdahl Bergersen
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from dolfin import *


# Skips with dependencies
skip_if_not_MPI = pytest.mark.skipif(not has_mpi(),
                                     reason="Skipping unit test(s) depending on MPI.")
skip_if_not_HDF5 = pytest.mark.skipif(not has_hdf5(),
                                      reason="Skipping unit test(s) depending on HDF5.")
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
# skip_64bit_int = pytest.mark.skipif(cpp.common.sizeof_la_index() == 8,
#                                     reason="This test does not work with 64-bit linear algebra indices.")

# Skips with respect to build type
skip_in_debug = pytest.mark.skipif(has_debug(),
                                   reason="This test does not work in debug mode.")
skip_in_release = pytest.mark.skipif(not has_debug(),
                                     reason="This test does not work in release mode.")
