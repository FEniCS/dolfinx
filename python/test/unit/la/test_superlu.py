# Copyright (C) 2026 Jack S. Hale, Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for MatrixCSR."""

from mpi4py import MPI

import dolfinx

import numpy as np
import pytest

@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="Not built with SuperLU_dist")
def test_superlu_solver():
    from dolfinx.la.superlu import superlu_solver
