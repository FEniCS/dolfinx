from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import dolfinx.cpp
from dolfinx.fem import (
    Function,
    FunctionSpace,
    create_sparsity_pattern,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.la import matrix_csr
from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver
from dolfinx.mesh import create_unit_square, locate_entities_boundary

# ruff: noqa
from ufl import TestFunction, TrialFunction, dx, grad, inner, sym, tr, Identity

mesh = create_unit_square(MPI.COMM_WORLD, 50, 50)
facets_bc = locate_entities_boundary(
    mesh,
    dim=mesh.topology.dim - 1,
    marker=lambda x: np.isclose(x[1], 0.0) & np.isclose(x[0], 0.5, 0.5),
)

facets_left = locate_entities_boundary(
    mesh, dim=(mesh.topology.dim - 1), marker=lambda x: np.isclose(x[0], 0.0)
)

facets_right = locate_entities_boundary(
    mesh, dim=(mesh.topology.dim - 1), marker=lambda x: np.isclose(x[0], 1.0)
)

V = functionspace(mesh, ("Lagrange", 1, (2,)))
dofsbc = locate_dofs_topological(V=V, entity_dim=1, entities=facets_bc)

dofsL = locate_dofs_topological(V=V, entity_dim=1, entities=facets_left)
dofsR = locate_dofs_topological(V=V, entity_dim=1, entities=facets_right)
coords = V.tabulate_dof_coordinates()

print(
    "V block size=",
    V.dofmap.bs,
    V.dofmap.index_map_bs,
    V.dofmap.list,
    V.dofmap.index_map.size_local,
)


ltog = V.dofmap.index_map.local_to_global(dofsR)
globalR = np.concatenate(mesh.comm.allgather(ltog))
globalR_coords = np.concatenate(mesh.comm.allgather(coords[dofsR]))


def cfun(p0, p1):
    """Find matching dofs on left and right side of the mesh.

    The right side is shifted by 1.0 in x-direction.
    """
    p1t = p1 + np.array([-1.0, 0, 0.0])
    if np.linalg.norm(p0 - p1t) < 1e-9:
        return True
    return False


# Creating mapping of left side to right side dofs
# using local index for left, global for right.
map_LR = {}
for dofL in dofsL:
    xL = coords[dofL]
    for dofR, xR in zip(globalR, globalR_coords):
        if cfun(xL, xR):
            map_LR[int(dofL) * 2] = (int(dofR * 2), 1.0)
            map_LR[int(dofL) * 2 + 1] = (int(dofR * 2 + 1), -1.0)

print(map_LR)

# Create MPC
local_dofs = np.array([k for k in map_LR.keys()], dtype=np.int32)
global_dofs = [np.array([map_LR[k][0]], dtype=np.int64) for k in map_LR.keys()]
global_coeffs = [np.array([map_LR[k][1]], dtype=np.float64) for k in map_LR.keys()]

print(local_dofs)

mpc = dolfinx.cpp.fem.MPC_float64(V._cpp_object, local_dofs, global_dofs, global_coeffs)
for cell in mpc.cells():
    dofs = mpc.V().dofmap.cell_dofs(cell)
    bs = mpc.V().dofmap.bs
    s = ""
    for d in dofs:
        if d * bs in local_dofs:
            s += f" {d * bs}*"
        else:
            s += f" {d * bs}"
    print(f"cell {cell} dofs {s}")

ufl_e = V.ufl_element()
V_new = FunctionSpace(mesh, ufl_e, mpc.V())

E = 100.0
ν = 0.3
μ = E / (2.0 * (1.0 + ν))
λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))


def σ(v):
    """Return an expression for the stress σ given a displacement field."""
    return 2.0 * μ * sym(grad(v)) + λ * tr(sym(grad(v))) * Identity(len(v))


u = TestFunction(V_new)
v = TrialFunction(V_new)
a = form(inner(σ(u), grad(v)) * dx)

f = Function(V_new)
f.interpolate(lambda x: [(x[0] - 0.1) ** 2, np.zeros_like(x[1])])
L = form(inner(f, v) * dx)

bc = dirichletbc(value=np.array([0.0, 0.0], dtype=np.float64), dofs=dofsbc, V=V_new)

# Create SparsityPattern
sp = create_sparsity_pattern(a)
# Add extra sparsity for MPC connections
dolfinx.cpp.fem.build_sparsity_pattern_mpc(sp, a._cpp_object, mpc)
sp.finalize()

from dolfinx.fem.petsc import create_matrix

A = create_matrix(a)
print(A)
A.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, True)

dolfinx.cpp.fem.petsc.assemble_matrix_mpc(mpc, A, a._cpp_object, [bc._cpp_object])
# dolfinx.fem.assemble_matrix(A, a, [bc])
A.assemble()
dolfinx.cpp.fem.petsc.insert_diagonal(A, a.function_spaces[0], [bc._cpp_object], 1.0)
A.assemble()


offsets, ref_dof, ref_coeff = mpc.constraints()
bs = V_new.dofmap.bs
print(len(offsets), offsets)
for i in range(V_new.dofmap.index_map.size_local * bs):
    if offsets[i + 1] - offsets[i] > 0:
        print(
            i,
            "is constrained to ",
            ref_dof[offsets[i] : offsets[i + 1]],
            "with coeffs",
            ref_coeff[offsets[i] : offsets[i + 1]],
        )

# Setting constraint b_i to zero
b = dolfinx.fem.assemble_vector(L)
dolfinx.fem.apply_lifting(b.array, [a], [[bc]])
b.scatter_reverse(dolfinx.la.InsertMode.add)
bc.set(b.array)

for i in range(V_new.dofmap.index_map.size_local * bs):
    if offsets[i + 1] - offsets[i] > 0:
        b.array[i] = 0.0

b.scatter_forward()

u = Function(V_new)

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
ksp.setFromOptions()
ksp.solve(b.petsc_vec, u.x.petsc_vec)

# Update ghost values, required since ref_dof (below) may index dofs
# owned by another rank when running in parallel
u.x.scatter_forward()

xdmf = dolfinx.io.XDMFFile(mesh.comm, "demo_mpc.xdmf", "w")
xdmf.write_mesh(mesh)
u.name = "u"
xdmf.write_function(u)

# Check that each constrained dof actually matches the linear combination
# of its reference dofs, i.e. u_i = sum_j coeff_j * u_{ref_j}
# Only owned dofs are checked, but reference dofs may be ghosts owned by
# another rank, so u.x must have up-to-date ghost values (scatter_forward
# above) for this to be valid when running in parallel.
for i in range(V_new.dofmap.index_map.size_local * bs):
    n0, n1 = offsets[i], offsets[i + 1]
    if n1 > n0:
        expected = np.dot(ref_coeff[n0:n1], u.x.array[ref_dof[n0:n1]])
        actual = u.x.array[i]
        assert np.isclose(actual, expected), (
            f"dof {i}: value {actual} does not match expected {expected} "
            f"from reference dofs {ref_dof[n0:n1]}"
        )
if mesh.comm.rank == 0:
    print("Constrained dofs match their reference values")
