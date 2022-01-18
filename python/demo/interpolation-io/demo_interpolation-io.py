
import numpy as np

from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace
from dolfinx.mesh import CellType, create_rectangle

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

# Create mesh
mesh = create_rectangle(MPI.COMM_WORLD, ((0.0, 0.0), (1.0, 1.0)), (16, 16), CellType.triangle)

# Create Nedelec function space and finite element Function
V = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", 2))
u = Function(V, dtype=ScalarType)


def f(x):
    """TODO"""
    v = x[:2, :]
    v[0] = np.where(x[0] < 0.5, v[0], v[0] + 1)
    return v


# Interpolate f in the Nedelec/H(curl) space
u.interpolate(f)

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


# # Plot solution
# try:
#     import pyvista
    # from dolfinx import plot
#     topology, cell_types = plot.create_vtk_topology(mesh, mesh.topology.dim)
#     grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
#     grid.point_data["u0"] = u0.real
#     grid.set_active_scalars("u0")

#     plotter = pyvista.Plotter()
#     plotter.add_mesh(grid, show_edges=True)
#     warped = grid.warp_by_scalar()
#     plotter.add_mesh(warped)
#     plotter.show()
# except ModuleNotFoundError:
#     print("pyvista is required to visualise the solution")
