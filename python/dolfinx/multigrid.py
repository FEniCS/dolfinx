# Copyright (C) 2024 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from dolfinx.cpp.la import SparsityPattern
from dolfinx.cpp.multigrid import assemble_transfer_matrix as _assemble_transfer_matrix
from dolfinx.cpp.multigrid import create_sparsity_pattern as _create_sparsity_pattern
from dolfinx.cpp.multigrid import inclusion_mapping as _inclusion_mapping
from dolfinx.fem import FunctionSpace
from dolfinx.la import MatrixCSR
from dolfinx.mesh import Mesh

__all__ = ["assemble_transfer_matrix", "create_sparsity_pattern", "inclusion_mapping"]


def inclusion_mapping(
    mesh_from: Mesh, mesh_to: Mesh, allow_all_to_all: bool = False
) -> NDArray[np.int64]:
    return _inclusion_mapping(mesh_from._cpp_object, mesh_to._cpp_object, allow_all_to_all)


def assemble_transfer_matrix(
    T: MatrixCSR,
    V_from: FunctionSpace,
    V_to: FunctionSpace,
    inclusion_map: NDArray[np.int64],
    weight: Callable[[int], float],
):
    _assemble_transfer_matrix(
        T._cpp_object, V_from._cpp_object, V_to._cpp_object, inclusion_map, weight
    )


def create_sparsity_pattern(
    V_from: FunctionSpace, V_to: FunctionSpace, inclusion_map: NDArray[np.int64]
) -> SparsityPattern:
    return _create_sparsity_pattern(V_from._cpp_object, V_to._cpp_object, inclusion_map)
