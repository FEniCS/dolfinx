# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Shared pytest fixtures for dolfinx.io unit tests."""

from mpi4py import MPI

import pytest

from dolfinx import default_real_type
from dolfinx.mesh import GhostMode, create_unit_cube, create_unit_interval, create_unit_square


@pytest.fixture
def mesh_factory():
    """Return a function building a unit mesh of a given topological dimension."""

    def _mesh_factory(tdim, n, ghost_mode=GhostMode.shared_facet, dtype=default_real_type):
        if tdim == 1:
            return create_unit_interval(MPI.COMM_WORLD, n, ghost_mode=ghost_mode, dtype=dtype)
        elif tdim == 2:
            return create_unit_square(MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode, dtype=dtype)
        elif tdim == 3:
            return create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=ghost_mode, dtype=dtype)
        else:
            raise ValueError(f"Unsupported topological dimension: {tdim}")

    return _mesh_factory
