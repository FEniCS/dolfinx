# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""FIXME: document"""

from dolfin import common
import ffc


def ffc_default_parameters():
    """Get default parameters of FFC"""
    # Get dict with defaults
    d = ffc.default_jit_parameters()
    typemap = {"quadrature_rule": "", "quadrature_degree": 0, "precision": 0}

    # Add the rest
    for key, value in d.items():
        if value is None:
            d.add(key, typemap[key])
            d[key] = None
        else:
            d.add(key, value)

    # Update the scalar type according to the mode (real or complex)
    d["scalar_type"] = "double complex" if common.has_petsc_complex else "double"

    return d
