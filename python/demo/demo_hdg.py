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
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_hdg.py>`
# * {download}`Jupyter notebook <./demo_hdg.ipynb>`
# ```
# This demo illustrates how to:
#
# - Solve Poisson's equation using an HDG scheme.
# - Defining custom integration domains
# - Create a submesh over all facets of the mesh
# - Use `ufl.MixedFunctionSpace` to defined blocked problems.
# - Assemble mixed systems with multiple, related meshes

# +
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import dolfinx
import ufl
from dolfinx import fem, mesh
from dolfinx.cpp.mesh import cell_num_entities
from dolfinx.fem import extract_function_spaces
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    assign,
    create_vector,
    set_bc,
)

# -


# We start by creating two convenience functions: One to compute
# the L2 norm of a UFL expression, and one to define the integration
# domains for the facets of all cells in a mesh.


def norm_L2(v: ufl.core.expr.Expr, measure: ufl.Measure = ufl.dx) -> np.inexact:
    """Convenience function to compute the L2 norm of a UFL expression."""
    compiled_form = fem.form(ufl.inner(v, v) * measure)
    comm = compiled_form.mesh.comm
    return np.sqrt(comm.allreduce(fem.assemble_scalar(compiled_form), op=MPI.SUM))


# In DOLFINx, we represent integration domains over entities of
# codimension > 0 as a tuple `(cell_idx, local_entity_idx)`,
# where `cell_idx` is the index of the cell in the mesh
# (local to process), and `local_entity_idx` is the local index
# of the sub entity (local to cell). For the HDG scheme, we will
# integrate over the facets of each cell (for internal facets,
# there will be repeat entries, from the viewpoint of the connected cells).


def compute_cell_boundary_facets(msh: dolfinx.mesh.Mesh) -> np.ndarray:
    """Compute the integration entities for integrals around the
    boundaries of all cells in msh.

    Parameters:
        msh: The mesh.

    Returns:
        Facets to integrate over, identified by ``(cell, local facet
        index)`` pairs.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    n_f = cell_num_entities(msh.topology.cell_type, fdim)
    n_c = msh.topology.index_map(tdim).size_local
    return np.vstack((np.repeat(np.arange(n_c), n_f), np.tile(np.arange(n_f), n_c))).T.flatten()


def u_e(x):
    """Exact solution."""
    u_e = 1
    for i in range(tdim):
        u_e *= ufl.sin(ufl.pi * x[i])
    return u_e


comm = MPI.COMM_WORLD
rank = comm.rank
dtype = PETSc.ScalarType

# Create the mesh

n = 8  # Number of elements in each direction
msh = mesh.create_unit_cube(comm, n, n, n, ghost_mode=mesh.GhostMode.none)

# We need to create a broken Lagrange space defined over the facets of
# the mesh. To do so, we require a sub-mesh of the all facets. We begin
# by creating a list of all of the facets in the mesh

tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_entities(fdim)
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
facets = np.arange(num_facets, dtype=np.int32)

# The submesh is created with {py:func}`dolfinx.mesh.create_submesh`,
# which takes in the mesh to extract entities from, the topological
# dimension of the entities, and the set of entities to create the
# submesh (indices local to process).
# ```{admonition} Note
# Despite all facets being present in the submesh, the entity map
# isn't necessarily the identity in parallel
# ```

facet_mesh, facet_mesh_emap = mesh.create_submesh(msh, fdim, facets)[:2]

# Define function spaces

k = 3  # Polynomial order
V = fem.functionspace(msh, ("Discontinuous Lagrange", k))
Vbar = fem.functionspace(facet_mesh, ("Discontinuous Lagrange", k))

# Trial and test functions in mixed space, we use {py:class}`
# ufl.MixedFunctionSpace`
# to create a single function space object we can extract {py:func}`
# ufl.TrialFunctions`
# and {py:func}`ufl.TestFunctions` from.

W = ufl.MixedFunctionSpace(V, Vbar)
u, ubar = ufl.TrialFunctions(W)
v, vbar = ufl.TestFunctions(W)


# ## Define integration measures

# We define the integration measure over cells as we would do in any
# other UFL form.

dx_c = ufl.Measure("dx", domain=msh)

# For the cell boundaries, we need to define an integration measure to
# integrate around the boundary of each cell. The integration entities
# can be computed using the following convenience function.

cell_boundary_facets = compute_cell_boundary_facets(msh)
cell_boundaries = 1  # A tag

# We pass the integration domains into the `ufl.Measure` through the
# `subdomain_data` keyword argument.
ds_c = ufl.Measure("ds", subdomain_data=[(cell_boundaries, cell_boundary_facets)], domain=msh)

# Create a cell integral measure over the facet mesh

dx_f = ufl.Measure("dx", domain=facet_mesh)

# ## Variational formulation

# +
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)
gamma = 16.0 * k**2 / h  # Scaled penalty parameter

x = ufl.SpatialCoordinate(msh)
c = 1.0 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
a = (
    ufl.inner(c * ufl.grad(u), ufl.grad(v)) * dx_c
    - ufl.inner(c * (u - ubar), ufl.dot(ufl.grad(v), n)) * ds_c(cell_boundaries)
    - ufl.inner(ufl.dot(ufl.grad(u), n), c * (v - vbar)) * ds_c(cell_boundaries)
    + gamma * ufl.inner(c * (u - ubar), v - vbar) * ds_c(cell_boundaries)
)

f = -ufl.div(c * ufl.grad(u_e(x)))  # Manufacture a source term
L = ufl.inner(f, v) * dx_c
L += ufl.inner(fem.Constant(facet_mesh, dtype(0.0)), vbar) * dx_f
# -

# Our bilinear form involves two domains (`msh` and `facet_mesh`). The
# mesh passed to the measure is called the "integration domain". For
# each additional mesh in our form, we must pass an
# {py:class}`EntityMap<dolfinx.mesh.EntityMap` object
# that relates entities in that mesh to entities in the integration
# domain. In this case, the only other mesh is `facet_mesh`, so we pass
# `facet_mesh_emap`.

entity_maps = [facet_mesh_emap]

# Compile forms for the blocked system, using {py:func}`ufl.extract_blocks`
# for the bilinear and linear forms.

a_blocked = dolfinx.fem.form(ufl.extract_blocks(a), entity_maps=entity_maps)
L_blocked = dolfinx.fem.form(ufl.extract_blocks(L))

# Apply Dirichlet boundary conditions. We begin by locating the boundary
# facets of msh.

msh_boundary_facets = mesh.exterior_facet_indices(msh.topology)

# Since the boundary condition is enforced in the facet space, we need
# to get the corresponding facets in `facet_mesh` using the entity map

facet_mesh_boundary_facets = facet_mesh_emap.sub_topology_to_topology(
    msh_boundary_facets, inverse=True
)

# Get the dofs and apply the boundary condition

facet_mesh.topology.create_connectivity(fdim, fdim)
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
bc = fem.dirichletbc(dtype(0.0), dofs, Vbar)

# Assemble the matrix and vector

A = assemble_matrix(a_blocked, bcs=[bc])
A.assemble()

b = assemble_vector(L_blocked)
bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a_blocked, 1), [bc])
apply_lifting(b, a_blocked, bcs=bcs1)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bcs0 = fem.bcs_by_block(extract_function_spaces(L_blocked), [bc])
set_bc(b, bcs0)

# Setup the solver

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Compute solution

try:
    x = create_vector([V, Vbar])
    ksp.solve(b, x)
    ksp.destroy()
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    b.destroy()
except PETSc.Error as e:  # type: ignore
    if e.ierr == 92:
        print("The required PETSc solver/preconditioner is not available. Exiting.")
        print(e)
        exit(0)
    else:
        raise e

# Create functions for the solution and update values

u, ubar = fem.Function(V, name="u"), fem.Function(Vbar, name="ubar")
assign(x, [u, ubar])
x.destroy()

# Write to file

if dolfinx.has_adios2:
    from dolfinx.io import VTXWriter

    with VTXWriter(msh.comm, "u.bp", u, "bp4") as f:
        f.write(0.0)
    with VTXWriter(msh.comm, "ubar.bp", ubar, "bp4") as f:
        f.write(0.0)
else:
    print("ADIOS2 required for VTX output")


# Compute errors

x = ufl.SpatialCoordinate(msh)
e_u = norm_L2(u - u_e(x))
x_bar = ufl.SpatialCoordinate(facet_mesh)
e_ubar = norm_L2(ubar - u_e(x_bar))
PETSc.Sys.Print(f"e_u = {e_u:.5e}")
PETSc.Sys.Print(f"e_ubar = {e_ubar:.5e}")
