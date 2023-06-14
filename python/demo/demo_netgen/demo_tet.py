from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import FunctionSpace, Function, Constant, petsc, form
from dolfinx.fem import locate_dofs_topological, dirichletbc, assemble_scalar
from dolfinx import default_scalar_type
from dolfinx.io import ngsio, XDMFFile

from ufl import TestFunction, TrialFunction, dot, grad, dx, inner
from petsc4py.PETSc import ScalarType
import numpy as np

import netgen.gui
from netgen.csg import CSGeometry, Pnt, OrthoBrick
geo = CSGeometry()
geo.Add(OrthoBrick(Pnt(0,0,0),Pnt(1,1,1)))

domain  = ngsio.model_to_mesh(geo, MPI.COMM_WORLD, hmax=0.1, gdim=3)
V = FunctionSpace(domain, ("CG", 3))
u = Function(V, dtype=default_scalar_type)
u.interpolate(lambda x: x[0]*x[1]*x[2])
integrand = form(u*dx)
print(assemble_scalar(integrand))

with XDMFFile(domain.comm, "XDMF/tet.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
