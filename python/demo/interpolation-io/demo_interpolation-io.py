
import numpy as np

from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace
from dolfinx.mesh import CellType, create_rectangle, locate_entities

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

# Create mesh
mesh = create_rectangle(MPI.COMM_WORLD, ((0.0, 0.0), (1.0, 1.0)), (16, 16), CellType.triangle)

# Create Nedelec function space and finite element Function
V = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", 2))
u = Function(V, dtype=ScalarType)

tdim = mesh.topology.dim
cells0 = locate_entities(mesh, tdim, lambda x: x[0] < 0.5)
cells1 = locate_entities(mesh, tdim, lambda x: x[0] >= 0.5)

# Interpolate a function in the Nedelec/H(curl) space
u.interpolate(lambda x: x[:2], cells0)
u.interpolate(lambda x: x[:2] + 1, cells1)

# Create a vector-valued discontinuous Lagrange space and function, and
# interpolate the H(curl) function u
V0 = VectorFunctionSpace(mesh, ("Discontinuous Lagrange", 1))
u0 = Function(V0, dtype=ScalarType)
u0.interpolate(u)

try:
    # Save the interpolated function u0 in VTX format
    from dolfinx.cpp.io import VTXWriter
    with VTXWriter(mesh.comm, "output_nedelec.bp", [u0._cpp_object]) as file:
        file.write(0.0)
except ImportError:
    print("ADIOS2 required for VTK output")
