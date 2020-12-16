# Copyright (C) 2017 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Linear algebra module for solving systems of equations"""

import numpy as np

from dolfinx import cpp
from dolfinx.la.solver import solve

__all__ = [
    "solve"
]

# Import pybind11 objects into dolfinx.la
from dolfinx.cpp.la import VectorSpaceBasis  # noqa
