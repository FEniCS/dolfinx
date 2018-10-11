# -*- coding: utf-8 -*-
# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Interfaces for compatibility with the legacy DOLFIN interface"""

from dolfin.function import functionspace


def FunctionSpace(mesh, element, degree=None):
    """Create a FunctionSpace from a mesh and an element"""
    if degree:
        return functionspace.FunctionSpace(mesh, (element, degree))
    else:
        return functionspace.FunctionSpace(mesh, element, degree)

