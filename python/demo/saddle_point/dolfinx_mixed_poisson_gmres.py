# ### Import necessary modules ###
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import basix
import ufl
import dolfinx
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

# ### Import mesh in dolfinx ###
gdim = 3
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD

nx, ny, nz = 20, 20, 20
mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD,
                               [[0.0, 0.0, 0.0],
                                [1., 1., 1.]],
                               [nx, ny, nz],
                               dolfinx.mesh.CellType.tetrahedron)

# ### Mark boundaries ###
# Identify entities with dimension boundary_dim
def z_0(x):
    return np.isclose(x[2], 0)


def y_0(x):
    return np.isclose(x[1], 0)


def x_0(x):
    return np.isclose(x[0], 0)

boundary_dim = mesh.topology.dim - 1
z_0_facets = dolfinx.mesh.locate_entities_boundary(mesh, boundary_dim, z_0)
y_0_facets = dolfinx.mesh.locate_entities_boundary(mesh, boundary_dim, y_0)
x_0_facets = dolfinx.mesh.locate_entities_boundary(mesh, boundary_dim, x_0)

# Boundary markers: x=0 is 30, y=0 is 18, z=0 is 1
marked_facets = np.hstack([z_0_facets, y_0_facets, x_0_facets])
marked_values = np.hstack([np.full_like(z_0_facets, 1),
                           np.full_like(y_0_facets, 18),
                           np.full_like(x_0_facets, 30)])
sorted_facets = np.argsort(marked_facets)
boundaries = dolfinx.mesh.meshtags(mesh,
                                   boundary_dim,
                                   marked_facets[sorted_facets],
                                   marked_values[sorted_facets])

# Save mesh and boundaries for visualisation
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh_data/mesh.xdmf",
                         "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(boundaries, mesh.geometry)

# ### Define function spaces ###
# Create RT1 and DG0 spaces
k = 1
Q_el = basix.ufl.element("RT", mesh.basix_cell(), k)
P_el = basix.ufl.element("DG", mesh.basix_cell(), k - 1)
V_el = basix.ufl.mixed_element([Q_el, P_el])
V = dolfinx.fem.FunctionSpace(mesh, V_el)

(sigma, u) = ufl.TrialFunctions(V)
(tau, v) = ufl.TestFunctions(V)

x = ufl.SpatialCoordinate(mesh)
# mu is parameter numpy array
mu = np.array([-2., 0.5, 0.5, 0.5, 3.])
f = 10. * ufl.exp(-mu[0] * ((x[0] - mu[1]) * (x[0] - mu[1]) +
                            (x[1] - mu[2]) * (x[1] - mu[2]) +
                            (x[2] - mu[3]) * (x[2] - mu[3])))

# Get subspace of V
V0 = V.sub(0)
Q, _ = V0.collapse()

dx = ufl.dx

# ### Weak formulation ###
a = ufl.inner(sigma, tau) * dx + \
    ufl.inner(u, ufl.div(tau)) * dx + \
    ufl.inner(ufl.div(sigma), v) * dx
L = -ufl.inner(f, v) * dx

# ### Boundary conditions ###
# Identify dofs on x=0 and define boundary condition functions
dofs_x0 = dolfinx.fem.locate_dofs_topological((V0, Q), gdim-1,
                                              boundaries.find(30))

def f1(x):
    values = np.zeros((3, x.shape[1]))
    values[0, :] = np.sin(mu[4] * x[0])
    return values


f_h1 = dolfinx.fem.Function(Q)
f_h1.interpolate(f1)
bc_x0 = dolfinx.fem.dirichletbc(f_h1, dofs_x0, V0)

# Identify dofs on y=0
dofs_y0 = dolfinx.fem.locate_dofs_topological((V0, Q), gdim-1,
                                              boundaries.find(18))


def f2(x):
    values = np.zeros((3, x.shape[1]))
    values[1, :] = np.sin(mu[4] * x[1])
    return values


f_h2 = dolfinx.fem.Function(Q)
f_h2.interpolate(f2)
bc_y0 = dolfinx.fem.dirichletbc(f_h2, dofs_y0, V0)

# Identify dofs on z=0
dofs_z0 = dolfinx.fem.locate_dofs_topological((V0, Q), gdim-1,
                                              boundaries.find(1))


def f3(x):
    values = np.zeros((3, x.shape[1]))
    values[2, :] = np.sin(mu[4] * x[2])
    return values


f_h3 = dolfinx.fem.Function(Q)
f_h3.interpolate(f3)
bc_z0 = dolfinx.fem.dirichletbc(f_h3, dofs_z0, V0)

# ### Compute solution field ###
# Gather boundary conditions
bcs = [bc_x0, bc_y0, bc_z0]

# Assemble the system matrices and the vectors
a_cpp = dolfinx.fem.form(a)
l_cpp = dolfinx.fem.form(L)
A = assemble_matrix(a_cpp, bcs=bcs)
A.assemble()
L = assemble_vector(l_cpp)
apply_lifting(L, [a_cpp], [bcs])
L.ghostUpdate(addv=PETSc.InsertMode.ADD,
              mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.petsc.set_bc(L, bcs)

# Solver setup
ksp = PETSc.KSP()
ksp.create(mesh.comm)
ksp.setOperators(A)
# Set GMRES solver
ksp.setType("gmres")
ksp.setGMRESRestart(100)
# Convergence criteria based on residual tolerance
ksp.rtol = 1.e-8
# Solve and see convergence details
ksp.setFromOptions()
w_h = dolfinx.fem.Function(V)
ksp.solve(L, w_h.vector)
print(f"Number of iterations: {ksp.getIterationNumber()}")
print(f"Convergence reason: {ksp.getConvergedReason()}")
ksp.destroy()
A.destroy()
L.destroy()
w_h.x.scatter_forward()
# Split the FEM solutions sigma and u
sigma_h, u_h = w_h.split()
sigma_h = sigma_h.collapse()
u_h = u_h.collapse()

# ### Save solution field for visualisation ###
with dolfinx.io.XDMFFile(mesh.comm, "out_mixed_poisson/sigma.xdmf",
                         "w") as sol_file_sigma:
    sol_file_sigma.write_mesh(mesh)
    sol_file_sigma.write_function(sigma_h)

with dolfinx.io.XDMFFile(mesh.comm, "out_mixed_poisson/u.xdmf",
                         "w") as sol_file_u:
    sol_file_u.write_mesh(mesh)
    sol_file_u.write_function(u_h)

sigma_norm = mesh.comm.allreduce(dolfinx.fem.assemble_scalar
                                 (dolfinx.fem.form(ufl.inner(sigma_h, sigma_h) *
                                                   dx +
                                                   ufl.inner(ufl.div(sigma_h),
                                                             ufl.div(sigma_h)) *
                                                             dx)), op=MPI.SUM)
u_norm = mesh.comm.allreduce(dolfinx.fem.assemble_scalar
                             (dolfinx.fem.form(ufl.inner(u_h, u_h) *
                                               dx)), op=MPI.SUM)

print(f"sigma norm: {sigma_norm}, u norm: {u_norm}")
