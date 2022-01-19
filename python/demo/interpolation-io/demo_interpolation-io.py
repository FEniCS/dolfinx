
# Copyright (C) 2022 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# ====================
# Interpolation and IO
# ====================
#
# This demo show the interpolation of functions into vector-element
# (H(curl)) finite element spaces, and the interpolation of these
# special finite elements in discontinuous Lagrange spaces for
# artifact-free visualisation.


import numpy as np

from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace
from dolfinx.mesh import CellType, create_rectangle, locate_entities

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

# Create a mesh. For what comes later in this demo we need to ensure
# that a boundary between cells is located at x0=0.5
mesh = create_rectangle(MPI.COMM_WORLD, ((0.0, 0.0), (1.0, 1.0)), (16, 16), CellType.triangle)

# Create Nedelec function space and finite element Function
V = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", 1))
u = Function(V, dtype=ScalarType)

# Find cells with *all* vertices (0) <= 0.5 or (1) >= 0.5
tdim = mesh.topology.dim
cells0 = locate_entities(mesh, tdim, lambda x: x[0] <= 0.5)
cells1 = locate_entities(mesh, tdim, lambda x: x[0] >= 0.5)

# Interpolate in the Nedelec/H(curl) space a vector-valued expression
# ``f``, where f \dot e_0 is discontinuous at x0 = 0.5 and  f \dot e_1
# is continuous.
u.interpolate(lambda x: np.vstack((x[0], x[1])), cells0)
u.interpolate(lambda x: np.vstack((x[0] + 1, x[1])), cells1)

# Create a vector-valued discontinuous Lagrange space and function, and
# interpolate the H(curl) function `u`
V0 = VectorFunctionSpace(mesh, ("Discontinuous Lagrange", 1))
u0 = Function(V0, dtype=ScalarType)
u0.interpolate(u)

try:
    # Save the interpolated function u0 in VTX format. It should be seen
    # when visualising that the x0-component is discontinuous across
    # x0=0.5 and the x0-component is continuous across x0=0.5
    from dolfinx.cpp.io import VTXWriter
    with VTXWriter(mesh.comm, "output_nedelec.bp", [u0._cpp_object]) as file:
        file.write(0.0)
except ImportError:
    print("ADIOS2 required for VTK output")
