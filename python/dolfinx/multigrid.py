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
    if np.issubdtype(mesh_from.geometry.x.dtype, np.float32):
        return _inclusion_mapping_float32(mesh_from._cpp_object, mesh_to._cpp_object)
    elif np.issubdtype(mesh_from.geometry.x.dtype, np.float64):
        return _inclusion_mapping_float64(mesh_from._cpp_object, mesh_to._cpp_object)
    else:
        raise RuntimeError("Unsupported mesh dtype.")
