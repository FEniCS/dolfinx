# Copyright (C) 2024 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
from numpy.typing import NDArray

from dolfinx.cpp.multigrid import inclusion_mapping as _inclusion_mapping
from dolfinx.mesh import Mesh

__all__ = ["inclusion_mapping"]


def inclusion_mapping(mesh_from: Mesh, mesh_to: Mesh) -> NDArray[np.int64]:
    return _inclusion_mapping(mesh_from._cpp_object, mesh_to._cpp_object)
