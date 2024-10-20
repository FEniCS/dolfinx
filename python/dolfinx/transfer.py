import numpy as np
from numpy.typing import NDArray

from dolfinx.cpp.la import SparsityPattern
from dolfinx.cpp.transfer import (
    assemble_transfer_matrix,
)
from dolfinx.cpp.transfer import create_sparsity_pattern as _create_sparsity_pattern
from dolfinx.cpp.transfer import inclusion_mapping as _inclusion_mapping
from dolfinx.fem import FunctionSpace
from dolfinx.mesh import Mesh

__all__ = ["inclusion_mapping", "create_sparsity_pattern", "assemble_transfer_matrix"]


def inclusion_mapping(mesh_from: Mesh, mesh_to: Mesh) -> NDArray[np.int64]:
    return _inclusion_mapping(mesh_from._cpp_object, mesh_to._cpp_object)


def create_sparsity_pattern(
    V_from: FunctionSpace, V_to: FunctionSpace, inclusion_map: NDArray[np.int64]
) -> SparsityPattern:
    return _create_sparsity_pattern(V_from._cpp_object, V_to._cpp_object, inclusion_map)
