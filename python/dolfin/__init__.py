# -*- coding: utf-8 -*-
"""Main module for DOLFIN"""

# flake8: noqa

# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# Distributed under the terms of the GNU Lesser Public License (LGPL),
# either version 3 of the License, or (at your option) any later
# version.

import sys

# Store dl open flags to restore them after import
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
                         timing, timings, list_timings, SubSystemsManager)

if has_hdf5():
    from .cpp.io import HDF5File

from .cpp import MPI
from .cpp.function import (Expression, Constant, FunctionAXPY,
                           LagrangeInterpolator)
from .cpp.fem import (FiniteElement, DofMap,
                      get_coordinates, create_mesh, set_coordinates,
                      vertex_to_dof_map, dof_to_vertex_map,
                      PointSource, DiscreteOperators,
                      SparsityPatternBuilder)

from .cpp.geometry import (BoundingBoxTree,
                           Point,
                           MeshPointIntersection,
                           intersect)
from .cpp.generation import IntervalMesh, BoxMesh, RectangleMesh
from .cpp.graph import GraphBuilder
from .cpp.io import XDMFFile, VTKFile
from .cpp.la import VectorSpaceBasis

from .cpp.la import (PETScVector, PETScMatrix,
                     PETScOptions, PETScLUSolver,
                     PETScKrylovSolver)
from .cpp.fem import PETScDMCollection

if has_slepc():
    from .cpp.la import SLEPcEigenSolver

from .cpp.la import (IndexMap, Scalar,
                     TensorLayout)
from .cpp.log import (info, Table, set_log_level, get_log_level, LogLevel)
from .cpp.math import ipow, near, between
from .cpp.mesh import (Mesh, MeshTopology, MeshGeometry, MeshEntity,
                       CellType, Cell, Facet, Face,
                       Edge, Vertex, cells, facets, faces, edges,
                       entities, vertices, SubDomain,
                       MeshEditor, MeshQuality,
                       PeriodicBoundaryComputation,
                       SubsetIterator)

from .cpp.nls import (NonlinearProblem, NewtonSolver, OptimisationProblem)
from .cpp.parameter import Parameters, parameters

# Import Python modules
from . import la
from . import mesh
from . import parameter

from .common import timer
from .common.timer import Timer, timed
from .common.plotting import plot

from .fem.assembling import (assemble_system,
                             SystemAssembler, assemble_local)
from .fem.form import Form
from .fem.norms import norm, errornorm
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

from .generation.builtin import UnitSquareMesh

# experimental
from .jit.pybind11jit import compile_cpp_code

from .la import as_backend_type, la_index_dtype
from .mesh.meshfunction import (MeshFunction)
from .mesh.meshvaluecollection import MeshValueCollection
from .mesh.subdomain import CompiledSubDomain

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
