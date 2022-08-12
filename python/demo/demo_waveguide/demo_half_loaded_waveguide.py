import numpy as np
from slepc4py import SLEPc

import ufl
from dolfinx import fem, io
from dolfinx.mesh import (CellType, create_rectangle, exterior_facet_indices,
                          locate_entities)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

l = 1
b = 0.5
d = 0.25

domain = create_rectangle(MPI.COMM_WORLD, [[0, 0], [l, b]], [
    80, 40], CellType.triangle)

domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

eps_v = 1
eps_d = 1


def Omega_d(x):
    return x[1] <= d


def Omega_v(x):
    return x[1] >= d


D = fem.FunctionSpace(domain, ("DG", 0))
eps = fem.Function(D)

cells_v = locate_entities(domain, domain.topology.dim, Omega_v)
cells_d = locate_entities(domain, domain.topology.dim, Omega_d)

eps.x.array[cells_d] = np.full_like(cells_d, eps_d, dtype=ScalarType)
eps.x.array[cells_v] = np.full_like(cells_v, eps_v, dtype=ScalarType)

c0 = 3 * 10**8  # m/s
MHz = 10**6
f0 = 350 * MHz
k0 = 2 * np.pi * c0 / f0

N1curl = ufl.FiniteElement("N1curl", domain.ufl_cell(), 3)
H1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 3)
V = fem.FunctionSpace(domain, ufl.MixedElement(N1curl, H1))

et, ez = ufl.TrialFunctions(V)
vt, vz = ufl.TestFunctions(V)

a_tt = (ufl.inner(ufl.curl(et), ufl.curl(vt)) - k0**2 * eps * ufl.inner(et, vt)) * ufl.dx
b_tt = ufl.inner(et, vt) * ufl.dx
b_tz = ufl.inner(et, ufl.grad(vz)) * ufl.dx
b_zt = ufl.inner(ufl.grad(ez), vt) * ufl.dx
b_zz = (ufl.inner(ufl.grad(ez), ufl.grad(vz)) - k0**2 * eps * ufl.inner(ez, vz)) * ufl.dx

a = fem.form(a_tt)
b = fem.form(b_tt + b_tz + b_zt + b_zz)

bc_facets = exterior_facet_indices(domain.topology)

bc_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, bc_facets)

u_bc = fem.Function(V)
with u_bc.vector.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)

A = fem.petsc.assemble_matrix(a, bcs=[bc])
A.assemble()
B = fem.petsc.assemble_matrix(b, bcs=[bc])
B.assemble()

eps = SLEPc.EPS().create(MPI.COMM_WORLD)
eps.setOperators(A, B)
#eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)
eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
#eps.setTarget(-k0**2)
#eps.setDimensions(nev=5)
eps.solve()

vals = [(i, eps.getEigenvalue(i)) for i in range(eps.getConverged()) if not
        (np.isclose(eps.getEigenvalue(i), 0))]

vals.sort(key=lambda x: x[1].real)

j = 0
E = fem.Function(V)

for i, _ in vals:

    E1, E2 = E.split()
    e = eps.getEigenpair(i, E.vector).real
    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
    print(f"error: {error}")
    print(f"eigenvalue: {np.sqrt(-eps.getEigenpair(i, E.vector))/k0:.12f}")
    E.name = f"E-{j:03d}-{eps.getEigenpair(i, E.vector).real:.4f}"
    j += 1

    E.x.scatter_forward()

    #V_dg = fem.VectorFunctionSpace(domain, ("DG", 3))
    #E_dg = fem.Function(V_dg)
    #E_dg.interpolate(E)
    #padded_j = str(j).zfill(3)
    #padded_e = str(e).zfill(3)
    #with io.VTXWriter(domain.comm, f"sols/E_{padded_j}_eigen{e:.0f}.bp", E_dg) as f:
    #    f.write(0.0)
