# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Main module for DOLFINX"""

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


from dolfinx.common import (has_debug, has_petsc_complex, has_kahip,
                           has_parmetis, git_commit_hash, TimingType, timing,
                           timings, list_timings)

import dolfinx.MPI
import dolfinx.log

from dolfinx.generation import (IntervalMesh, BoxMesh, RectangleMesh,
                               UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh)

from .cpp.mesh import Mesh, Topology, Geometry, MeshEntity, MeshQuality

from .cpp.nls import (NonlinearProblem, NewtonSolver)

from .fem.form import Form
from .fem.dirichletbc import DirichletBC
from .fem.solving import solve

from .function import (FunctionSpace, VectorFunctionSpace,
                       TensorFunctionSpace, Constant, Function)
from .specialfunctions import (FacetNormal, CellVolume, CellNormal,
                               CellDiameter, Circumradius)

from .mesh import MeshTags

# Initialise PETSc and logging
from dolfinx import cpp
import sys
# FIXME: We're not passing command link argument here because some
# pytest arg crash loguru
cpp.common.SubSystemsManager.init_logging([""])
# cpp.common.SubSystemsManager.init_logging(sys.argv)
del sys
cpp.common.SubSystemsManager.init_petsc()

def get_include(user=False):
    import os
    d = os.path.dirname(__file__)
    if os.path.exists(os.path.join(d, "include")):
        # Package is installed
        return os.path.join(d, "include")
    else:
        # Package is from a source directory
        return os.path.join(os.path.dirname(d), "src")