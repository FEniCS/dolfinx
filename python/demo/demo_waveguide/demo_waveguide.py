import dolfinx
import dolfinx.io
import ufl
from mpi4py import MPI
from slepc4py import SLEPc
import numpy as np

domain = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [np.pi, np.pi]], [100, 100], dolfinx.mesh.CellType.triangle)
domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)

N1curl = ufl.FiniteElement("N1curl", domain.ufl_cell(), 2)
V = dolfinx.fem.FunctionSpace(domain, N1curl)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = dolfinx.fem.form(ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx)
b = dolfinx.fem.form(ufl.inner(u, v) * ufl.dx)

bc_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)

bc_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim-1, bc_facets)

u_bc = dolfinx.fem.Function(V)
with u_bc.vector.localForm() as loc:
    loc.set(0)
bc = dolfinx.fem.dirichletbc(u_bc, bc_dofs)

A = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
A.assemble()
B = dolfinx.fem.petsc.assemble_matrix(b, bcs=[bc])
B.assemble()

eps = SLEPc.EPS().create(MPI.COMM_WORLD)
eps.setOperators(A, B)
eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)
eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eps.setTarget(5.5)
eps.setDimensions(40)
eps.solve()

vals = [(i, eps.getEigenvalue(i)) for i in range(eps.getConverged()) if not
        (np.isclose(eps.getEigenvalue(i), 1) or np.isclose(eps.getEigenvalue(i), 0))]

vals.sort(key=lambda x: x[1].real)

j = 0
E = dolfinx.fem.Function(V)

for i, _ in vals:
    
    e = eps.getEigenpair(i, E.vector).real
    print(f"eigenvalue: {eps.getEigenpair(i, E.vector).real:.12f}")
    E.name = f"E-{j:03d}-{eps.getEigenpair(i, E.vector).real:.4f}"
    j += 1

    E.x.scatter_forward()

    V_dg = dolfinx.fem.VectorFunctionSpace(domain, ("DG", 2))
    E_dg = dolfinx.fem.Function(V_dg)
    E_dg.interpolate(E)
    padded_j = str(j).zfill(3)
    padded_e = str(e).zfill(3)
    with dolfinx.io.VTXWriter(domain.comm, f"sols/E_{padded_j}_eigen{e:.0f}.bp", E_dg) as f:
        f.write(0.0)