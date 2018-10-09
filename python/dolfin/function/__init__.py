# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin.function.argument import (TestFunction, TrialFunction, Argument,
                                      TestFunctions, TrialFunctions)
from dolfin.function.constant import Constant
from dolfin.function.expression import (BaseExpression, CompiledExpression,
                                        Expression)
from dolfin.function.function import Function
from dolfin.function.functionspace import (FunctionSpace, VectorFunctionSpace,
                                           TensorFunctionSpace)
from dolfin.function.jit import compile_expression, jit_generate
from dolfin.function.specialfunctions import (
    MeshCoordinates, FacetArea, CellDiameter, CellNormal, CellVolume,
    Circumradius, SpatialCoordinate, MinCellEdgeLength, MaxCellEdgeLength,
    MinFacetEdgeLength, MaxFacetEdgeLength, FacetNormal)

__all__ = [
    "TestFunction", "TrialFunction", "Argument", "TestFunctions",
    "TrialFunctions", "Constant", "BaseExpression", "CompiledExpression",
    "Expression", "Function", "FunctionSpace", "VectorFunctionSpace",
    "TensorFunctionSpace", "compile_expression", "jit_generate",
    "MeshCoordinates", "FacetArea", "CellDiameter", "CellNormal", "CellVolume",
    "Circumradius", "SpatialCoordinate", "MinCellEdgeLength",
    "MaxCellEdgeLength", "MinFacetEdgeLength", "MaxFacetEdgeLength",
    "FacetNormal"
]
