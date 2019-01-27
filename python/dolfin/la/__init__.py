# -*- coding: utf-8 -*-
# Copyright (C) 2017 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

from dolfin import cpp
from dolfin.la.solver import solve

__all__ = [
    "solve"
]

# Import pybind11 objects into dolfin.la
from dolfin.cpp.la import VectorSpaceBasis  # noqa
from dolfin.cpp.la import (PETScKrylovSolver, PETScOptions)  # noqa


def la_index_dtype():
    """Return the numpy dtype equivalent to the type of la_index"""
    return np.intc if cpp.common.sizeof_la_index() == 4 else np.int64
