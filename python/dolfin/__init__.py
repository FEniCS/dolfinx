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
from .cpp.common import (Variable, has_debug, has_hdf5, has_scotch,
                         has_hdf5_parallel, has_mpi, has_petsc,
                         has_parmetis, has_slepc, git_commit_hash,
                         DOLFIN_EPS, DOLFIN_PI, TimingClear,
                         TimingType, timing, timings, list_timings,
                         dump_timings_to_xml)

if has_hdf5():
    from .cpp.adaptivity import TimeSeries
    from .cpp.io import HDF5File

from .cpp.ale import ALE
from .cpp import MPI
from .cpp.function import (Expression, Constant, FunctionAXPY,
                           LagrangeInterpolator, FunctionAssigner,
                           assign, MultiMeshFunction, MultiMeshFunctionSpace)
from .cpp.fem import (FiniteElement, DofMap, Assembler,
                      get_coordinates, create_mesh, set_coordinates,
                      vertex_to_dof_map, dof_to_vertex_map,
                      PointSource, DiscreteOperators,
                      LinearVariationalSolver,
                      NonlinearVariationalSolver,
                      SparsityPatternBuilder)

from .cpp.geometry import (BoundingBoxTree,
                           Point,
                           MeshPointIntersection,
                           intersect,
                           CollisionPredicates,
                           IntersectionConstruction)
from .cpp.generation import (IntervalMesh, BoxMesh, RectangleMesh,
                             UnitDiscMesh, UnitQuadMesh, UnitHexMesh,
                             UnitTriangleMesh, UnitCubeMesh,
                             UnitSquareMesh, UnitIntervalMesh,
                             SphericalShellMesh)
from .cpp.graph import GraphBuilder
from .cpp.io import File, XDMFFile, VTKFile
from .cpp.la import (has_linear_algebra_backend,
                     linear_algebra_backends,
                     has_krylov_solver_method,
                     has_krylov_solver_preconditioner, normalize,
                     VectorSpaceBasis, in_nullspace)

if has_linear_algebra_backend('PETSc'):
    from .cpp.la import (PETScVector, PETScMatrix, PETScFactory,
                         PETScOptions, PETScLUSolver,
                         PETScKrylovSolver, PETScPreconditioner)
    from .cpp.fem import PETScDMCollection
    from .cpp.nls import (PETScSNESSolver, PETScTAOSolver, TAOLinearBoundSolver)

if has_linear_algebra_backend('Tpetra'):
    from .cpp.la import (TpetraVector, TpetraMatrix, TpetraFactory,
                         MueluPreconditioner, BelosKrylovSolver)

if has_slepc():
    from .cpp.la import SLEPcEigenSolver

from .cpp.la import (IndexMap, DefaultFactory, Matrix, Vector, Scalar,
                     EigenMatrix, EigenVector, EigenFactory, LUSolver,
                     KrylovSolver, TensorLayout, LinearOperator,
                     BlockMatrix, BlockVector)
from .cpp.la import GenericVector  # Remove when pybind11 transition complete
from .cpp.log import (info, Table, set_log_level, get_log_level, LogLevel)
from .cpp.math import ipow, near, between
from .cpp.mesh import (Mesh, MeshTopology, MeshGeometry, MeshEntity,
                       MeshColoring, CellType, Cell, Facet, Face,
                       Edge, Vertex, cells, facets, faces, edges,
                       entities, vertices, SubDomain, BoundaryMesh,
                       MeshEditor, MeshQuality, SubMesh,
                       DomainBoundary, PeriodicBoundaryComputation,
                       MeshTransformation, SubsetIterator, MultiMesh)

from .cpp.nls import (NonlinearProblem, NewtonSolver, OptimisationProblem)
from .cpp.refinement import refine
from .cpp.parameter import Parameters, parameters
from .cpp.io import X3DOM, X3DOMParameters

# Import Python modules
from . import io
from . import la
from . import mesh
from . import parameter

from .common import timer
from .common.timer import Timer, timed
from .common.plotting import plot

from .fem.assembling import (assemble, assemble_system,
                             SystemAssembler, assemble_local)
from .fem.form import Form
from .fem.norms import norm, errornorm
from .fem.dirichletbc import DirichletBC, AutoSubDomain
from .fem.interpolation import interpolate
from .fem.projection import project
from .fem.solving import (solve, LocalSolver,
                          LinearVariationalProblem,
                          NonlinearVariationalProblem)
from .fem.formmanipulations import (derivative, adjoint, increase_order, tear)

# Need to be careful with other to avoid circular dependency
from .fem.adaptivesolving import (AdaptiveLinearVariationalSolver,
                                  AdaptiveNonlinearVariationalSolver)

from .function.functionspace import (FunctionSpace,
                                     VectorFunctionSpace, TensorFunctionSpace)
from .function.function import Function
from .function.argument import (TestFunction, TrialFunction,
                                TestFunctions, TrialFunctions)
from .function.constant import Constant
from .function.specialfunctions import (FacetNormal, CellSize,
                                        SpatialCoordinate, CellVolume,
                                        Circumradius, FacetArea,
                                        MeshCoordinates)
from .function.expression import Expression, UserExpression, CompiledExpression

# experimental
from .jit.pybind11jit import compile_cpp_code

from .la import as_backend_type, la_index_dtype
from .mesh.ale import (compute_vertex_map, compute_edge_map,
                       init_parent_edge_indices)
from .mesh.meshfunction import (MeshFunction, CellFunction,
                                FacetFunction, FaceFunction,
                                EdgeFunction, VertexFunction)
from .mesh.meshvaluecollection import MeshValueCollection
from .mesh.subdomain import CompiledSubDomain

from .multistage.multistagescheme import (RK4, CN2, ExplicitMidPoint,
                                          ESDIRK3, ESDIRK4,
                                          ForwardEuler, BackwardEuler)
from .multistage.multistagesolvers import PointIntegralSolver, RKSolver
from .multistage.rushlarsenschemes import RL1, RL2, GRL1, GRL2

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


# FIXME
def has_petsc4py():
    return False


# FIXME: remove after transition
def has_pybind11():
    return True


# FIXME: remove after transition
def mpi_comm_self():
    return MPI.comm_self


# FIXME: remove after transition
def mpi_comm_world():
    return MPI.comm_world

# FIXME: remove all these after transition
TimingClear_clear = TimingClear.clear
TimingClear_keep = TimingClear.keep
TimingType_wall = TimingType.wall
TimingType_system = TimingType.system
TimingType_user = TimingType.user

IndexMap.MapSize_OWNED = IndexMap.MapSize.OWNED
IndexMap.MapSize_UNOWNED = IndexMap.MapSize.UNOWNED
IndexMap.MapSize_ALL = IndexMap.MapSize.ALL

TensorLayout.Sparsity_DENSE = TensorLayout.Sparsity.DENSE
TensorLayout.Sparsity_SPARSE = TensorLayout.Sparsity.SPARSE
TensorLayout.Ghosts_GHOSTED = TensorLayout.Ghosts.GHOSTED
TensorLayout.Ghosts_UNGHOSTED = TensorLayout.Ghosts.UNGHOSTED

if has_linear_algebra_backend('PETSc'):
    PETScKrylovSolver.norm_type_default_norm = PETScKrylovSolver.norm_type.default_norm
    PETScKrylovSolver.norm_type_natural = PETScKrylovSolver.norm_type.natural
    PETScKrylovSolver.norm_type_preconditioned = PETScKrylovSolver.norm_type.preconditioned
    PETScKrylovSolver.norm_type_none = PETScKrylovSolver.norm_type.none
    PETScKrylovSolver.norm_type_unpreconditioned = PETScKrylovSolver.norm_type.unpreconditioned

LocalSolver.SolverType_LU = LocalSolver.SolverType.LU
LocalSolver.SolverType_Cholesky = LocalSolver.SolverType.Cholesky

XDMFFile.Encoding_HDF5 = XDMFFile.Encoding.HDF5
XDMFFile.Encoding_ASCII = XDMFFile.Encoding.ASCII
