# Copyright (C) 2026 Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Registry from (scalar dtype, geometry dtype) to compiled C++ type.

``Function``, ``Expression``, ``Form`` and ``DirichletBC`` each own a
static ``_cpp_registry`` dict (bound to the class) mapping a
``(scalar, geometry)`` dtype pair to the compiled C++ type or factory.
DOLFINx registers the matched-precision pairs; downstream libraries add
further pairs (e.g. fp16 dofs on an fp64 mesh) with
:func:`register_cpp_type`.
"""

import numpy as np


def register_cpp_type(cls, scalar_dtype, geometry_dtype, cpp_type):
    """Register ``cpp_type`` on ``cls`` for a (scalar, geometry) pair."""
    cls._cpp_registry[np.dtype(scalar_dtype), np.dtype(geometry_dtype)] = cpp_type


def get_cpp_type(cls, scalar_dtype, geometry_dtype):
    """Compiled C++ type registered on ``cls`` for a dtype pair."""
    key = np.dtype(scalar_dtype), np.dtype(geometry_dtype)
    try:
        return cls._cpp_registry[key]
    except KeyError:
        raise NotImplementedError(
            f"No compiled {cls.__name__} for scalar '{key[0]}' on geometry "
            f"'{key[1]}'. Register one with dolfinx.fem.register_cpp_type()."
        ) from None


# Matched-precision built-ins (geometry == real part of scalar) as
# (scalar, geometry, C++ suffix).
MATCHED_PRECISIONS = (
    (np.float32, np.float32, "float32"),
    (np.float64, np.float64, "float64"),
    (np.complex64, np.float32, "complex64"),
    (np.complex128, np.float64, "complex128"),
)
