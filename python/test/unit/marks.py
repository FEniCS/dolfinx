# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Parametrize marks shared by the unit tests.

`dtype`, `cell_type` and `ghost_mode` have a default set applied by
`pytest_generate_tests` in conftest.py, so a test that wants the default does
not need a mark at all. Use these for anything else, e.g.

    from unit.marks import simplex_cells

    @simplex_cells
    def test_foo(cell_type): ...
"""

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.mesh import CellType, GhostMode

all_dtypes = pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])

# Tests that compile generated code with complex scalars cannot run on win32,
# whose compiler has no C99 _Complex. Tests that only move complex numbers
# around are fine there and use `all_dtypes`.
_complex_win32_xfail = [
    pytest.param(np.complex64, marks=pytest.mark.xfail_win32_complex),
    pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
]
all_dtypes_win32_xfail = pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, *_complex_win32_xfail]
)

all_cells_with_interval = pytest.mark.parametrize(
    "cell_type",
    [
        CellType.interval,
        CellType.triangle,
        CellType.quadrilateral,
        CellType.tetrahedron,
        CellType.hexahedron,
    ],
)
simplex_cells = pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.tetrahedron])
tp_cells = pytest.mark.parametrize("cell_type", [CellType.quadrilateral, CellType.hexahedron])
cells_2d = pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.quadrilateral])
cells_3d = pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])

# Interior facet integrals need ghosted facets to be correct in parallel.
interior_facet_ghost_modes = pytest.mark.parametrize(
    "ghost_mode",
    [
        pytest.param(
            GhostMode.none,
            marks=pytest.mark.skipif(
                condition=MPI.COMM_WORLD.size > 1,
                reason="Unghosted interior facets fail in parallel",
            ),
        ),
        GhostMode.shared_facet,
    ],
)
