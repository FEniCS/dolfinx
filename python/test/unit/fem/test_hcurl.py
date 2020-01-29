# Copyright (C) 2020 Igor A Baratta
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import pytest

import dolfinx
import ufl


@pytest.mark.parametrize("space", ["N1curl", "N2curl"])
@pytest.mark.parametrize("degree", [1,
                                    pytest.param(2, marks=pytest.mark.xfail),
                                    pytest.param(3, marks=pytest.mark.xfail)])
def test_assembly_hcurl(space, degree):
    """ Manufactured solution for the curl-curl problem with homogeneous
    Dirichlet boundary condition on the tangencial trace.
    """
    pygmsh = pytest.importorskip("pygmsh")

    if dolfinx.MPI.rank(dolfinx.MPI.comm_world) == 0:
        geom = pygmsh.built_in.Geometry()
        geom.add_rectangle(0., 1., 0., 1., 0., lcar=0.01)
        mesh = pygmsh.generate_mesh(geom, verbose=True)
        mesh.prune()
        points, cells = mesh.points, mesh.cells
    else:
        points = numpy.zeros([0, 3])
        cells = {"triangle": numpy.zeros([0, 3], dtype=numpy.int64)}

    mesh = dolfinx.Mesh(dolfinx.MPI.comm_world, dolfinx.cpp.mesh.CellType.triangle, points[:, :2],
                        cells['triangle'], [], dolfinx.cpp.mesh.GhostMode.none)
    cmap = dolfinx.fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap

    k = 1
    alpha = (2 * numpy.pi**2 + k)

    def expr(x):
        return numpy.stack([numpy.cos(numpy.pi * x[0]) * numpy.sin(numpy.pi * x[1]),
                           -numpy.sin(numpy.pi * x[0]) * numpy.cos(numpy.pi * x[1])],
                           axis=0)

    V = dolfinx.FunctionSpace(mesh, (space, degree))
    W = dolfinx.VectorFunctionSpace(mesh, ("CG", degree + 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dim = mesh.topology.dim
    boundaries = numpy.where(mesh.topology.on_boundary(dim - 1))[0]
    u0 = dolfinx.Function(V)
    u0.vector.set(0.0)
    bc = dolfinx.DirichletBC(V, u0, boundaries)

    # Interpolate exact solution
    u_ex = dolfinx.Function(W)
    u_ex.interpolate(expr)

    a = ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx + k * ufl.inner(u, v) * ufl.dx
    L = alpha * ufl.inner(u_ex, v) * ufl.dx

    # Compute solution
    u = dolfinx.Function(V)
    dolfinx.solve(a == L, u, bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # Compute L2 error
    e = dolfinx.fem.assemble_scalar(ufl.inner(u - u_ex, u - u_ex) * ufl.dx)
    assert abs(e) < 1e-2
