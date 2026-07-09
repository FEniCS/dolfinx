# Copyright (C) 2026 Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Registry from (kind, scalar dtype, geometry dtype) to compiled type.

``Function``, ``Expression``, ``Form`` and ``DirichletBC`` are templated
on both a scalar and a mesh geometry type. DOLFINx registers the
matched-precision pairs; downstream libraries register further pairs
(e.g. fp16 dofs on an fp64 mesh) via :func:`register_cpp_type`.
"""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt

from dolfinx import cpp as _cpp

__all__ = ["get_cpp_type", "register_cpp_type"]

# Object kinds templated on (scalar, geometry).
_KINDS = ("Function", "Expression", "Form", "DirichletBC")

# (kind, scalar dtype, geometry dtype) -> compiled C++ class or factory.
_registry: dict[tuple[str, np.dtype, np.dtype], typing.Any] = {}


def register_cpp_type(
    kind: str,
    scalar_dtype: npt.DTypeLike,
    geometry_dtype: npt.DTypeLike,
    cpp_type: typing.Any,
) -> None:
    """Register the compiled C++ type for an object kind and dtype pair.

    Args:
        kind: Object kind, one of ``"Function"``, ``"Expression"``,
            ``"Form"`` or ``"DirichletBC"``.
        scalar_dtype: Scalar (degree-of-freedom) type.
        geometry_dtype: Mesh geometry type.
        cpp_type: Compiled C++ class or factory that the wrapper builds
            with for this pair.
    """
    if kind not in _KINDS:
        raise ValueError(f"Unknown object kind '{kind}'. Expected one of {_KINDS}.")
    _registry[(kind, np.dtype(scalar_dtype), np.dtype(geometry_dtype))] = cpp_type


def get_cpp_type(
    kind: str, scalar_dtype: npt.DTypeLike, geometry_dtype: npt.DTypeLike
) -> typing.Any:
    """Look up the compiled C++ type for an object kind and dtype pair.

    Args:
        kind: Object kind (see :func:`register_cpp_type`).
        scalar_dtype: Scalar (degree-of-freedom) type.
        geometry_dtype: Mesh geometry type.

    Returns:
        The registered C++ class or factory.
    """
    scalar, geometry = np.dtype(scalar_dtype), np.dtype(geometry_dtype)
    try:
        return _registry[(kind, scalar, geometry)]
    except KeyError:
        raise NotImplementedError(
            f"No {kind} type registered for scalar dtype '{scalar}' on geometry "
            f"dtype '{geometry}'. Downstream libraries can register one with "
            f"dolfinx.fem.register_cpp_type()."
        ) from None


# Register the matched-precision built-ins (geometry == real part of
# scalar). ``Expression`` is built through a factory, the others through
# their class constructor.
_MATCHED: tuple[tuple[npt.DTypeLike, npt.DTypeLike, str], ...] = (
    (np.float32, np.float32, "float32"),
    (np.float64, np.float64, "float64"),
    (np.complex64, np.float32, "complex64"),
    (np.complex128, np.float64, "complex128"),
)

for _scalar, _geom, _name in _MATCHED:
    register_cpp_type("Function", _scalar, _geom, getattr(_cpp.fem, f"Function_{_name}"))
    register_cpp_type("Expression", _scalar, _geom, getattr(_cpp.fem, f"create_expression_{_name}"))
    register_cpp_type("Form", _scalar, _geom, getattr(_cpp.fem, f"Form_{_name}"))
    register_cpp_type("DirichletBC", _scalar, _geom, getattr(_cpp.fem, f"DirichletBC_{_name}"))
