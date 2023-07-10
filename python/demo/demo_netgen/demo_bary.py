from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form
from dolfinx.io import XDMFFile, ngsio
from mpi4py import MPI
from netgen.geom2d import SplineGeometry
from petsc4py import PETSc
from ufl import dx

from dolfinx import default_scalar_type

# Creating Netgen Geometry
geo = SplineGeometry()
geo.AddRectangle((0, 0), (1, 1))
# Setting up a PETSc Transform
tr = PETSc.DMPlexTransform().create(comm=PETSc.COMM_WORLD)
tr.setType(PETSc.DMPlexTransformType.REFINEALFELD)
# Construct DOLFINx mesh
domain = ngsio.model_to_mesh(
    geo, MPI.COMM_WORLD, hmax=0.1, gdim=2, transform=tr)
V = FunctionSpace(domain, ("Lagrange", 3))
u = Function(V, dtype=default_scalar_type)
u.interpolate(lambda x: x[0] * x[1])
integrand = form(u * dx)
print(assemble_scalar(integrand))

with XDMFFile(domain.comm, "XDMF/bary.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
