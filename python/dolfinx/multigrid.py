# Copyright (C) 2025 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
from numpy.typing import NDArray

from dolfinx.cpp.multigrid import inclusion_mapping_float32 as _inclusion_mapping_float32
from dolfinx.cpp.multigrid import inclusion_mapping_float64 as _inclusion_mapping_float64
from dolfinx.mesh import Mesh

__all__ = ["inclusion_mapping"]


def inclusion_mapping(mesh_from: Mesh, mesh_to: Mesh) -> NDArray[np.int64]:
    """Computes an inclusion map: a map between vertex indices from one
    mesh to another.

    Args:
        mesh_from: Domain of the map.
        mesh_to: Range of the map.

    Returns:
        Inclusion map, the `i`-th component is the vertex index of the
        vertex with the same coordinates in `mesh_to` and `-1` if it can
        not be found (locally!) in `mesh_to`. If `map[i] != -1` it holds
        `mesh_from.geometry.x[i] == mesh_to.geometry.x[map[i]]`.

    Note:
        Invoking `inclusion_map` on a `(mesh_coarse, mesh_fine)` tuple,
        where `mesh_fine` is produced by refinement with
        `IdentityPartitionerPlaceholder()`option, the returned `map` is
        guaranteed to match all vertices for all locally owned vertices
        (not for the ghost vertices).
    """
    if np.issubdtype(mesh_from.geometry.x.dtype, np.float32):
        return _inclusion_mapping_float32(mesh_from._cpp_object, mesh_to._cpp_object)
    elif np.issubdtype(mesh_from.geometry.x.dtype, np.float64):
        return _inclusion_mapping_float64(mesh_from._cpp_object, mesh_to._cpp_object)
    else:
        raise RuntimeError("Unsupported mesh dtype.")
