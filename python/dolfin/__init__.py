# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Main module for DOLFIN"""

# flake8: noqa

# Store dl open flags to restore them after import
import sys
stored_dlopen_flags = sys.getdlopenflags()

# Developer note: below is related to OpenMPI
# Fix dlopen flags (may need reorganising)
if "linux" in sys.platform:
    # FIXME: What with other platforms?
    try:
        from ctypes import RTLD_NOW, RTLD_GLOBAL
    except ImportError:
        RTLD_NOW = 2
        RTLD_GLOBAL = 256
    sys.setdlopenflags(RTLD_NOW | RTLD_GLOBAL)
del sys

# Reset dl open flags
# sys.setdlopenflags(stored_dlopen_flags)
# del sys

# Import cpp modules
from .cpp import __version__


from dolfin.common import (TimingType, git_commit_hash, has_debug,
                           has_parmetis, has_petsc_complex, list_timings,
                           timing, timings)


import dolfin.MPI

from dolfin.fem import DofMap
from dolfin.generation import BoxMesh, IntervalMesh, RectangleMesh
from dolfin.geometry import BoundingBoxTree, Point



from .cpp.mesh import (Cell, CellRange, Cells, CellType, Edge, EdgeRange,
                       Edges, EntityRange, Face, FaceRange, Faces, Facet,
                       FacetRange, Facets, Geometry, Mesh, MeshEntities,
                       MeshEntity, MeshQuality, PeriodicBoundaryComputation,
                       SubDomain, Topology, Vertex, VertexRange, Vertices)


from .cpp.nls import (NonlinearProblem, NewtonSolver)

from .fem.form import Form
from .fem.dirichletbc import DirichletBC
from .fem.interpolation import interpolate
from .fem.projection import project
from .fem.solving import solve
from .fem.formmanipulations import adjoint, derivative, increase_order, tear

from .function.functionspace import (FunctionSpace, VectorFunctionSpace,
                                     TensorFunctionSpace)
from .function.function import Function
from .function.argument import (TestFunction, TrialFunction, TestFunctions,
                                TrialFunctions)
from .function.specialfunctions import (CellDiameter, CellNormal, CellVolume,
                                        Circumradius, FacetNormal,
                                        MaxCellEdgeLength, MaxFacetEdgeLength,
                                        MinCellEdgeLength, MinFacetEdgeLength,
                                        SpatialCoordinate)

from .function.expression import Expression

from .generation import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh

# experimental
from .pybind11jit import compile_cpp_code

#from .la import la_index_dtype
from .mesh import MeshFunction
from .mesh import MeshValueCollection

# Import from ufl
from ufl import (Dx, FiniteElement, Identity, Measure, MixedElement,
                 NodalEnrichedElement, TensorElement, VectorElement, acos,
                 as_matrix, as_tensor, as_vector, asin, atan, avg, bessel_I,
                 bessel_J, bessel_K, bessel_Y, conditional, conj, cos, cross,
                 curl, dC, det, dev, diff, div, dot, dP, ds, dS, dx, dX,
                 elem_div, elem_mult, elem_op, elem_pow, erf, exp, ge, grad,
                 gt, hexahedron, imag, inner, interval, jump, le, lhs, ln, lt,
                 outer, pi, quadrilateral, real, rhs, sin, skew, split, sqrt,
                 sym, system, tan, tetrahedron, tr, triangle, variable)
from ufl.formoperators import action

# Initialise PETSc
from dolfin import cpp
cpp.common.SubSystemsManager.init_petsc()
