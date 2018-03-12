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

from .cpp.common import (Variable, has_debug, has_hdf5, has_scotch,
                         has_hdf5_parallel, has_mpi, has_mpi4py,
                         has_petsc, has_petsc4py, has_parmetis,
                         has_slepc, has_slepc4py, git_commit_hash,
                         DOLFIN_EPS, DOLFIN_PI, TimingClear, TimingType,
                         timing, timings, list_timings)

if has_hdf5():
    from .cpp.io import HDF5File

from .cpp import MPI
from .cpp.function import (Expression, Constant, FunctionAXPY)
from .cpp.fem import (FiniteElement, DofMap,
                      vertex_to_dof_map, dof_to_vertex_map)

from .cpp.geometry import BoundingBoxTree, Point
from .cpp.generation import IntervalMesh, BoxMesh, RectangleMesh
from .cpp.io import XDMFFile, VTKFile

if has_slepc():
    from .cpp.la import SLEPcEigenSolver

from .cpp.mesh import (Mesh, MeshTopology, MeshGeometry,
                       CellType,
                       Cell, Facet, Face, Edge, Vertex, MeshEntity,
                       Cells, Facets, Faces, Edges, Vertices, MeshEntities,
                       CellRange, FacetRange, FaceRange, EdgeRange, VertexRange, EntityRange,
                       SubDomain, MeshQuality,
                       PeriodicBoundaryComputation)

from .cpp.nls import (NonlinearProblem, NewtonSolver, OptimisationProblem)
#from .cpp.parameter import Parameters, parameters

# Import Python modules
#from . import la
# from . import mesh
#from . import parameter

#from .common import timer
#from .common.timer import Timer, timed

# from .fem.assembling import (assemble_system,
#                             SystemAssembler, assemble_local)

from .fem.form import Form
from .fem.dirichletbc import DirichletBC, AutoSubDomain
from .fem.interpolation import interpolate
from .fem.projection import project
from .fem.solving import solve
from .fem.formmanipulations import (derivative, adjoint, increase_order, tear)

from .function.functionspace import (FunctionSpace,
                                     VectorFunctionSpace, TensorFunctionSpace)
from .function.function import Function
from .function.argument import (TestFunction, TrialFunction,
                                TestFunctions, TrialFunctions)
from .function.constant import Constant
from .function.specialfunctions import (MeshCoordinates, FacetArea, FacetNormal,
                                        CellVolume, SpatialCoordinate, CellNormal,
                                        CellDiameter, Circumradius,
                                        MinCellEdgeLength, MaxCellEdgeLength,
                                        MinFacetEdgeLength, MaxFacetEdgeLength)
from .function.expression import Expression, UserExpression, CompiledExpression

from .generation import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh

# experimental
from .jit.pybind11jit import compile_cpp_code

#from .la import la_index_dtype
from .mesh import MeshFunction
from .mesh import MeshValueCollection

# Import from ufl
from ufl import (FiniteElement, TensorElement, VectorElement,
                 MixedElement, rhs, lhs, conditional, le, lt, ge, gt,
                 split, cross, inner, dot, grad, curl, dx, div,
                 Measure, det, pi, sin, cos, tan, acos, asin, atan,
                 ln, exp, sqrt, bessel_I, bessel_J, bessel_K,
                 bessel_Y, Dx, ds, dS, dP, dX, dC, interval, triangle,
                 tetrahedron, quadrilateral, hexahedron, avg, jump,
                 sym, tr, Identity, variable, diff, as_vector,
                 as_tensor, as_matrix, system, outer, dev, skew,
                 elem_mult, elem_div, elem_pow, elem_op, erf)
from ufl.formoperators import action


# Initialise PETSc
cpp.common.SubSystemsManager.init_petsc()
