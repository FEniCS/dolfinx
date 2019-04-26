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


from dolfin.common import (has_debug, has_petsc_complex,
                           has_parmetis, git_commit_hash, TimingType,
                           timing, timings, list_timings)

import dolfin.MPI

from dolfin.geometry import  Point
from dolfin.generation import (IntervalMesh, BoxMesh, RectangleMesh,
                               UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh)

from .cpp.mesh import (Mesh, Topology, Geometry, CellType, Cell, Facet,
                       Face, Edge, Vertex, MeshEntity, Cells, Facets, Faces,
                       Edges, Vertices, MeshEntities, CellRange, FacetRange,
                       FaceRange, EdgeRange, VertexRange, EntityRange,
                       SubDomain, MeshQuality)

from .cpp.nls import (NonlinearProblem, NewtonSolver)

from .fem.form import Form
from .fem.dirichletbc import DirichletBC
from .fem.interpolation import interpolate
from .fem.projection import project
from .fem.solving import solve

from .function.functionspace import (FunctionSpace, VectorFunctionSpace,
                                     TensorFunctionSpace)
from .function.function import Function
from .function.argument import (TestFunction, TrialFunction, TestFunctions,
                                TrialFunctions)
from .function.specialfunctions import (FacetNormal, CellVolume, CellNormal, CellDiameter, Circumradius)
from .function.expression import Expression

from .mesh import MeshFunction
from .mesh import MeshValueCollection


# Initialise PETSc
from dolfin import cpp
cpp.common.SubSystemsManager.init_petsc()
