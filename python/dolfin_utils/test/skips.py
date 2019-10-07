# Copyright (C) 2014-2014 Aslak Wigdahl Bergersen
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Shared skips for unit tests involving dolfin."""

import pytest

from dolfin import MPI
from dolfin.common import has_petsc_complex

# Skips with respect to parallel or serial
xfail_in_parallel = pytest.mark.xfail(
    MPI.size(MPI.comm_world) > 1,
    reason="This test does not yet work in parallel.")
skip_in_parallel = pytest.mark.skipif(
    MPI.size(MPI.comm_world) > 1,
    reason="This test should only be run in serial.")

# Skips with respect to the scalar type
skip_if_complex = pytest.mark.skipif(
    has_petsc_complex, reason="This test does not work in complex mode.")
xfail_if_complex = pytest.mark.xfail(
    has_petsc_complex, reason="This test does not work in complex mode.")
