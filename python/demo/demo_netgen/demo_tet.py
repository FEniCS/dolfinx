from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form
from dolfinx.io import XDMFFile, ngsio
from mpi4py import MPI
from netgen.csg import CSGeometry, OrthoBrick, Pnt
from ufl import dx

from dolfinx import default_scalar_type

geo = CSGeometry()
geo.Add(OrthoBrick(Pnt(0, 0, 0), Pnt(1, 1, 1)))

domain = ngsio.model_to_mesh(geo, MPI.COMM_WORLD, hmax=0.1, gdim=3)
V = FunctionSpace(domain, ("Lagrange", 3))
u = Function(V, dtype=default_scalar_type)
u.interpolate(lambda x: x[0] * x[1] * x[2])
integrand = form(u * dx)
print(assemble_scalar(integrand))

with XDMFFile(domain.comm, "XDMF/tet.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
