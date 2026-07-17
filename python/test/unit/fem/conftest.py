# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Shared pytest fixtures for dolfinx.fem unit tests."""

import math

import pytest


@pytest.fixture
def nest_matrix_norm():
    """Return a function computing the norm of a PETSc MatNest matrix."""

    def _nest_matrix_norm(A):
        assert A.getType() == "nest"
        norm = 0.0
        nrows, ncols = A.getNestSize()
        for row in range(nrows):
            for col in range(ncols):
                A_sub = A.getNestSubMatrix(row, col)
                if A_sub:
                    _norm = A_sub.norm()
                    norm += _norm * _norm
        return math.sqrt(norm)

    return _nest_matrix_norm
