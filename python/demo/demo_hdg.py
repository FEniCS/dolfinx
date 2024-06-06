# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # HDG scheme for the Poisson equation
#
# This demo is implemented in {download}`demo_hdg.py`. It
# illustrates how to:
#
# - Solve Poisson's equation using an HDG scheme.

# +
import importlib.util

if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
    from petsc4py import PETSc

    from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block

else:
    print("This demo requires petsc4py.")
    exit(0)

import sys

from mpi4py import MPI

import numpy as np

import ufl
from dolfinx import fem, mesh
from dolfinx.cpp.mesh import cell_num_entities
from ufl import div, dot, grad, inner


def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def norm_L2(comm, v, measure=ufl.dx):
    return np.sqrt(
        comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(v, v) * measure)), op=MPI.SUM)
    )


def compute_cell_boundary_facets(msh):
    """
    Compute the integration entities for integrals around the
    boundaries of all cells in msh.

    Parameters:
        msh: the mesh

    Returns:
        A list of facets to integrate over, identified by
        (cell, local facet index) pairs
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    n_f = cell_num_entities(msh.topology.cell_type, fdim)
    n_c = msh.topology.index_map(tdim).size_local
    cell_boundary_facets = np.vstack(
        (np.repeat(np.arange(n_c), n_f), np.tile(np.arange(n_f), n_c))
    ).T.flatten()
    return cell_boundary_facets


# Exact solution
def u_e(x):
    u_e = 1
    for i in range(tdim):
        u_e *= ufl.sin(ufl.pi * x[i])
    return u_e


# Boundary marker
def boundary(x):
    lr = np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
    tb = np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
    lrtb = lr | tb
    if tdim == 2:
        return lrtb
    else:
        assert tdim == 3
        fb = np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0)
        return lrtb | fb


comm = MPI.COMM_WORLD
rank = comm.rank
dtype = PETSc.ScalarType

# Number of elements in each direction
n = 8

# Create the mesh
msh = mesh.create_unit_cube(comm, n, n, n, ghost_mode=mesh.GhostMode.none)

# We need to create a broken Lagrange space defined over the facets of the
# mesh. To do so, we require a sub-mesh of the all facets. We begin by
# creating a list of all of the facets in the mesh
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_entities(fdim)
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
facets = np.arange(num_facets, dtype=np.int32)

# Create the sub-mesh
# NOTE Despite all facets being present in the submesh, the entity map isn't
# necessarily the identity in parallel
facet_mesh, facet_mesh_to_mesh = mesh.create_submesh(msh, fdim, facets)[:2]

# Define function spaces
k = 3  # Polynomial order
V = fem.functionspace(msh, ("Discontinuous Lagrange", k))
Vbar = fem.functionspace(facet_mesh, ("Discontinuous Lagrange", k))

# Trial and test functions
# Cell space
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
# Facet space
ubar, vbar = ufl.TrialFunction(Vbar), ufl.TestFunction(Vbar)

# Define integration measures
# Cell
dx_c = ufl.Measure("dx", domain=msh)
# Cell boundaries
# We need to define an integration measure to integrate around the
# boundary of each cell. The integration entities can be computed
# using the following convenience function.
cell_boundary_facets = compute_cell_boundary_facets(msh)
cell_boundaries = 1  # A tag
# Create the measure
ds_c = ufl.Measure("ds", subdomain_data=[(cell_boundaries, cell_boundary_facets)], domain=msh)
# Create a cell integral measure over the facet mesh
dx_f = ufl.Measure("dx", domain=facet_mesh)

# We write the mixed domain forms as integrals over msh. Hence, we must
# provide a map from facets in msh to cells in facet_mesh. This is the
# 'inverse' of facet_mesh_to_mesh, which we compute as follows:
mesh_to_facet_mesh = np.full(num_facets, -1)
mesh_to_facet_mesh[facet_mesh_to_mesh] = np.arange(len(facet_mesh_to_mesh))
entity_maps = {facet_mesh: mesh_to_facet_mesh}

# Define forms
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)
gamma = 16.0 * k**2 / h  # Scaled penalty parameter

x = ufl.SpatialCoordinate(msh)
c = 1.0 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
a_00 = fem.form(
    inner(c * grad(u), grad(v)) * dx_c
    - (
        inner(c * u, dot(grad(v), n)) * ds_c(cell_boundaries)
        + inner(c * v, dot(grad(u), n)) * ds_c(cell_boundaries)
    )
    + gamma * inner(c * u, v) * ds_c(cell_boundaries)
)
a_10 = fem.form(
    inner(dot(grad(u), n) - gamma * u, c * vbar) * ds_c(cell_boundaries), entity_maps=entity_maps
)
a_01 = fem.form(
    inner(dot(grad(v), n) - gamma * v, c * ubar) * ds_c(cell_boundaries), entity_maps=entity_maps
)
a_11 = fem.form(gamma * inner(c * ubar, vbar) * ds_c(cell_boundaries), entity_maps=entity_maps)

# Manufacture a source term
f = -div(c * grad(u_e(x)))

L_0 = fem.form(inner(f, v) * dx_c)
L_1 = fem.form(inner(fem.Constant(facet_mesh, dtype(0.0)), vbar) * dx_f)

# Define block structure
a = [[a_00, a_01], [a_10, a_11]]
L = [L_0, L_1]

# Apply Dirichlet boundary conditions
# We begin by locating the boundary facets of msh
msh_boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary)
# Since the boundary condition is enforced in the facet space, we must
# use the mesh_to_facet_mesh map to get the corresponding facets in
# facet_mesh
facet_mesh_boundary_facets = mesh_to_facet_mesh[msh_boundary_facets]
# Get the dofs and apply the bondary condition
facet_mesh.topology.create_connectivity(fdim, fdim)
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
bc = fem.dirichletbc(dtype(0.0), dofs, Vbar)

# Assemble the matrix and vector
A = assemble_matrix_block(a, bcs=[bc])
A.assemble()
b = assemble_vector_block(L, a, bcs=[bc])

# Setup the solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Compute solution
x = A.createVecRight()
ksp.solve(b, x)

# Create functions for the solution and update values
u, ubar = fem.Function(V), fem.Function(Vbar)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u.x.array[:offset] = x.array_r[:offset]
ubar.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
u.x.scatter_forward()
ubar.x.scatter_forward()

# Write to file
try:
    from dolfinx.io import VTXWriter

    with VTXWriter(msh.comm, "u.bp", u, "bp4") as f:
        f.write(0.0)
    with VTXWriter(msh.comm, "ubar.bp", ubar, "bp4") as f:
        f.write(0.0)
except ImportError:
    print("ADIOS2 required for VTX output")


# Compute errors
x = ufl.SpatialCoordinate(msh)
e_u = norm_L2(msh.comm, u - u_e(x))
x_bar = ufl.SpatialCoordinate(facet_mesh)
e_ubar = norm_L2(msh.comm, ubar - u_e(x_bar))
par_print(comm, f"e_u = {e_u}")
par_print(comm, f"e_ubar = {e_ubar}")
