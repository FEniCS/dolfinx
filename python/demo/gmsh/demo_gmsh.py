# Demonstrate the creation of meshes using pygmsh

import pygmsh
from mpi4py import MPI
import numpy as np

import ufl
# from dolfinx import fem, cpp
from dolfinx.io import XDMFFile
from dolfinx.mesh import create as create_mesh

cell = ufl.Cell("tetrahedron", geometric_dimension=3)
domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
if MPI.COMM_WORLD.rank == 0:
    geom = pygmsh.opencascade.Geometry()
    geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    pygmsh_mesh = pygmsh.generate_mesh(geom)
    pygmsh_mesh.prune()
    mesh = create_mesh(MPI.COMM_WORLD, pygmsh_mesh.cells[0].data, pygmsh_mesh.points, domain)
    points, cells = pygmsh_mesh.points, pygmsh_mesh.cells
else:
    mesh = create_mesh(MPI.COMM_WORLD, np.empty([0, 0]), np.empty([0, 3]), domain)


with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
    file.write_mesh(mesh)
