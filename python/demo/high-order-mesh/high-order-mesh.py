# Copyright (C) 2019 Jørgen Schartum Dokken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# This demo illustrates how to generate higher order meshes uses pygmsh
# and how to load them into dolfin

# Higher-order mesh with VTK visualization
# ========================================

from pygmsh.built_in import Geometry
from pygmsh import generate_mesh

import meshio
import numpy as np

from dolfin import Mesh, cpp, MPI, Constant
from dolfin.io import VTKFile
from dolfin.fem import assemble_scalar
from ufl import dx

# Generate high order gmsh
for order in range(1, 4):
    # Generate an annulus with inner radius r and outer radius R
    # Using pygmsh
    lcar = 1  # Mesh resolution in GMSH
    geo = Geometry()
    r = 0.5   # Small radius
    R = 1     # Big radius
    geo.add_raw_code("Mesh.ElementOrder={0};".format(order))
    small_circle = geo.add_circle([0, 0, 0], r, lcar=lcar)
    circle = geo.add_circle([0, 0, 0], R, lcar=lcar, holes=[small_circle.plane_surface])
    geo.add_physical(circle.plane_surface, 1)

    # Generate 2D mesh
    pygmsh = generate_mesh(geo, prune_z_0=True, verbose=False)

    if order == 1:
        element = "triangle"
    else:
        element = "triangle{0:d}".format(int((order + 1) * (order + 2) / 2))

    if order <= 2:
        # Write initial meshio mesh (only supports CG1 and CG2 meshes
        meshio.write("mesh_meshio_order{0}.xdmf".format(order),
                     meshio.Mesh(points=pygmsh.points,
                                 cells={element: pygmsh.cells[element]}),
                     file_format="xdmf")

    # Load points and connectivities into dolfin
    mesh = Mesh(MPI.comm_world, cpp.mesh.CellType.triangle, pygmsh.points,
                pygmsh.cells[element], [], cpp.mesh.GhostMode.none)

    # Solve Poisson problem
    from ufl import inner, grad, dx, ds, SpatialCoordinate, sqrt
    from dolfin import DirichletBC, Function, FunctionSpace, solve, TrialFunction, TestFunction
    from dolfin.fem import create_coordinate_map
    cmap = create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap
    V = FunctionSpace(mesh, ("CG", 1))
    u, v = TrialFunction(V), TestFunction(V)
    f = Constant(mesh, 1)
    a = inner(grad(u), grad(v))*dx
    x, y = SpatialCoordinate(mesh)
    g = -0.5*sqrt(x**2 + y**2)
    L = inner(f, v)*dx + inner(g, v) * ds
    # Boundary Condition
    u_D = Function(V)
    u_D.vector.set(0.0)
    def boundary(x, only_boundary):
        return np.sqrt(x[:,0]**2 + x[:,1]**2) > R - 1e-14
    bc = DirichletBC(V, u_D, boundary)

    uh = Function(V)
    solve(a==L, uh, bc)

    VTKFile("u{0}.pvd".format(order)).write(uh)
    int_u = MPI.sum(mesh.mpi_comm(), assemble_scalar(uh*dx))
    int_ex = 0.0520833
    print(int_u, int_ex, abs(int_u-int_ex)/(int_ex)*100)
    # Compute area of mesh
    area = MPI.sum(mesh.mpi_comm(), assemble_scalar(Constant(mesh, 1) * dx))

    area_ex = np.pi * (R**2 - r**2)
    print("-" * 20 + "Mesh order {0:d}".format(order) + "-" * 20)
    print("Computed volume: {0:.2e}"
          .format(area))
    print("Percentage offset: {0:.2e}".format(100 * abs(area - area_ex) / area))

    # Write mesh to VTK
    VTKFile("mesh_order{0}.pvd".format(order)).write(mesh)
