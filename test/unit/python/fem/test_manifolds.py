#!/usr/bin/env py.test

"""Unit tests for the solve function on manifolds
embedded in higher dimensional spaces."""

# Copyright (C) 2012 Imperial College London and others.
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by David Ham 2012

# MER: The solving test should be moved into test/regression/..., the
# evaluatebasis part should be moved into test/unit/FiniteElement.py

import pytest
from dolfin import *
from six.moves import zip
from six.moves import xrange as range
import os
import numpy
from dolfin_utils.test import *


# Subdomain to extract bottom boundary.
class BottomEdge(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 0.0)


class Rotation:
    """Class implementing rotations of the unit plane through an angle
    of phi about the x axis followed by theta about the z axis."""
    def __init__(self, phi, theta):
        self.theta = theta
        self.mat = numpy.dot(self._zmat(theta), self._xmat(phi))
        self.invmat = numpy.dot(self._xmat(-phi), self._zmat(-theta))

    def _zmat(self, theta):
        return numpy.array([[numpy.cos(theta), -numpy.sin(theta), 0.0],
                            [numpy.sin(theta),  numpy.cos(theta), 0.0],
                            [0.0,           0.0,          1.0]])

    def _xmat(self, phi):
        return numpy.array([[1.0,           0.0,           0.0],
                            [0.0,  numpy.cos(phi), -numpy.sin(phi)],
                            [0.0,  numpy.sin(phi),  numpy.cos(phi)]])

    def to_plane(self, x):
        """Map the point x back to the horizontal plane."""
        return numpy.dot(self.invmat, x)

    def x(self, i):
        """Produce a C expression for the ith component
        of the image of x mapped back to the horizontal plane."""

        return "("+" + ".join(["%.17f * x[%d]" % (a, j)
                               for (j, a) in enumerate(self.invmat[i, :])])+")"

    def rotate(self, mesh):
        """Rotate mesh through phi then theta."""

        mesh.coordinates()[:, :] = \
            numpy.dot(mesh.coordinates()[:, :], self.mat.T)

    def rotate_point(self, point):
        """Rotate point through phi then theta."""

        return numpy.dot(point, self.mat.T)


def poisson_2d():

    # Create mesh and define function space
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    g = Expression("sin(5*x[0])", degree=2)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    return u


def poisson_manifold():

    # Create mesh
    cubemesh = UnitCubeMesh(32, 32, 2)
    boundarymesh = BoundaryMesh(cubemesh, "exterior")
    mesh = SubMesh(boundarymesh, BottomEdge())

    rotation = Rotation(numpy.pi/4, numpy.pi/4)
    rotation.rotate(mesh)

    # Define function space
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        return rotation.to_plane(x)[0] < DOLFIN_EPS or \
            rotation.to_plane(x)[0] > 1.0 - DOLFIN_EPS

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression(("10*exp(-(pow(x[0] - %.17f, 2)" +
                    " + pow(x[1] - %.17f, 2)" +
                    " + pow(x[2] - %.17f, 2)) / 0.02)")
                   % tuple(rotation.rotate_point([0.5, 0.5, 0])), degree=2)
    g = Expression("sin(5*%s)" % rotation.x(0), degree=2)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    return u


def rotate_2d_mesh(theta):
    """Unit square mesh in 2D rotated through theta about the x and z
    axes."""

    cubemesh = UnitCubeMesh(1, 1, 1)
    boundarymesh = BoundaryMesh(cubemesh, "exterior")
    mesh = SubMesh(boundarymesh, BottomEdge())

    mesh.init_cell_orientations(Expression(("0.", "0.", "1."), degree=0))

    rotation = Rotation(theta, theta)
    rotation.rotate(mesh)

    return mesh


@skip_in_parallel
def test_poisson2D_in_3D():
    """This test solves Poisson's equation on a unit square in 2D, and
    then on a unit square embedded in 3D and rotated pi/4 radians
    about each of the z and x axes.

    """

    u_2D = poisson_2d()
    u_manifold = poisson_manifold()

    assert round(u_2D.vector().norm("l2") - u_manifold.vector().norm("l2"),
                 10) == 0
    assert round(u_2D.vector().max() - u_manifold.vector().max(), 10) == 0
    assert round(u_2D.vector().min() - u_manifold.vector().min(), 10) == 0


# TODO: Use pytest parameterization
@skip_in_parallel
def test_basis_evaluation_2D_in_3D():
    """This test checks that basis functions and their derivatives are
    unaffected by rotations."""

    basemesh = rotate_2d_mesh(0.0)
    rotmesh = rotate_2d_mesh(numpy.pi/4)
    rotation = Rotation(numpy.pi/4, numpy.pi/4)

    for i in range(4):
        basis_test("CG", i + 1, basemesh, rotmesh, rotation)
    for i in range(5):
        basis_test("DG", i, basemesh, rotmesh, rotation)
    for i in range(4):
        basis_test("RT", i + 1, basemesh, rotmesh, rotation, piola=True,)
    for i in range(4):
        basis_test("DRT", i + 1, basemesh, rotmesh, rotation, piola=True)
    for i in range(4):
        basis_test("BDM", i + 1, basemesh, rotmesh, rotation, piola=True)
    for i in range(4):
        basis_test("N1curl", i + 1, basemesh, rotmesh, rotation, piola=True)
        basis_test("BDFM", 2, basemesh, rotmesh, rotation, piola=True)


def basis_test(family, degree, basemesh, rotmesh, rotation, piola=False):
    ffc_option = "no-evaluate_basis_derivatives"
    basis_derivatives = parameters["form_compiler"][ffc_option]
    parameters["form_compiler"]["no-evaluate_basis_derivatives"] = False

    f_base = FunctionSpace(basemesh, family, degree)
    f_rot = FunctionSpace(rotmesh, family, degree)

    points = numpy.array([[1.0, 1.0, 0.0],
                          [0.5, 0.5, 0.0],
                          [0.3, 0.7, 0.0],
                          [0.4, 0.0, 0.0]])

    for cell_base, cell_rot in zip(cells(basemesh), cells(rotmesh)):

        values_base = numpy.zeros(f_base.element().value_dimension(0))
        derivs_base = numpy.zeros(f_base.element().value_dimension(0)*3)
        values_rot = numpy.zeros(f_rot.element().value_dimension(0))
        derivs_rot = numpy.zeros(f_rot.element().value_dimension(0)*3)

        # Get cell vertices
        vertex_coordinates_base = cell_base.get_vertex_coordinates()
        vertex_coordinates_rot  = cell_rot.get_vertex_coordinates()

        for i in range(f_base.element().space_dimension()):
            for point in points:
                f_base.element().evaluate_basis(i, values_base,
                                                point,
                                                vertex_coordinates_base,
                                                cell_base.orientation())

                f_base.element().evaluate_basis_derivatives(i, 1, derivs_base,
                                                            point,
                                                            vertex_coordinates_base,
                                                            cell_base.orientation())

                f_rot.element().evaluate_basis(i, values_rot,
                                                rotation.rotate_point(point),
                                                vertex_coordinates_rot,
                                                cell_rot.orientation())

                f_base.element().evaluate_basis_derivatives(i, 1, derivs_rot,
                                                            rotation.rotate_point(point),
                                                            vertex_coordinates_rot,
                                                            cell_rot.orientation())

                if piola:
                    values_cmp = rotation.rotate_point(values_base)

                    derivs_rot2 = derivs_rot.reshape(f_rot.element().value_dimension(0),3)
                    derivs_base2 = derivs_base.reshape(f_base.element().value_dimension(0),3)
                    # If D is the unrotated derivative tensor, then
                    # RDR^T is the rotated version.
                    derivs_cmp = numpy.dot(rotation.mat,
                                            rotation.rotate_point(derivs_base2))
                else:
                    values_cmp = values_base
                    # Rotate the derivative for comparison.
                    derivs_cmp = rotation.rotate_point(derivs_base)
                    derivs_rot2 = derivs_rot

                assert round(abs(derivs_rot2-derivs_cmp).max() - 0.0, 10) == 0
                assert round(abs(values_cmp-values_rot).max() - 0.0, 10) == 0

    parameters["form_compiler"]["no-evaluate_basis_derivatives"] = basis_derivatives


@use_gc_barrier
def test_elliptic_eqn_on_intersecting_surface(datadir):
    """Solves -grad^2 u + u = f on domain of two intersecting square
     surfaces embedded in 3D with natural bcs. Test passes if at end
     \int u dx = \int f dx over whole domain

    """
    # This needs to be odd
    #num_vertices_side = 31
    #mesh = make_mesh(num_vertices_side)
    #file = File("intersecting_surfaces.xml.gz", "compressed")
    #file << mesh
    filename = os.path.join(datadir, "intersecting_surfaces.xml")
    mesh = Mesh(filename)

    # function space, etc
    V = FunctionSpace(mesh, "CG", 2)
    u = TrialFunction(V)
    v = TestFunction(V)

    class Source(Expression):
        def eval(self, value, x):
            # r0 should be less than 0.5 * sqrt(2) in order for source to be
            # exactly zero on vertical part of domain
            r0 = 0.7
            r = sqrt(x[0] * x[0] + x[1] * x[1])
            if r < r0:
                value[0] = 20.0 * pow((r0 - r), 2)
            else:
                value[0] = 0.0

    f = Function(V)
    f.interpolate(Source(degree=2))

    a = inner(grad(u), grad(v))*dx + u*v*dx
    L = f*v*dx

    u = Function(V)
    solve(a == L, u)

    f_tot = assemble(f*dx)
    u_tot = assemble(u*dx)

    # test passes if f_tot = u_tot
    assert abs(f_tot - u_tot) < 1e-7


def make_mesh(num_vertices_side):
    # each square has unit side length
    domain_size = 1.0

    center_index = (num_vertices_side - 1) / 2
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, 2, 3)

    num_vertices = 2 * num_vertices_side * num_vertices_side - center_index - 1
    num_cells = 4 * (num_vertices_side - 1) * (num_vertices_side - 1)

    editor.init_vertices(num_vertices)
    editor.init_cells(num_cells)

    spacing = domain_size / (num_vertices_side - 1.0)

    # array of vertex indices
    v = [[0]*num_vertices_side for i in range(num_vertices_side)]

    # horizontal part of domain vertices
    vertex_count = 0

    for i in range(num_vertices_side):
        y = i * spacing
        for j in range(num_vertices_side):
            x = j * spacing
            p = Point(x, y, 0.0)
            editor.add_vertex(vertex_count, p)
            v[i][j] = vertex_count
            vertex_count += 1

    # cells
    cell_count = 0
    for i in range(num_vertices_side - 1):
        for j in range(num_vertices_side - 1):
            editor.add_cell(cell_count, v[i][j], v[i][j+1], v[i+1][j])
            cell_count += 1

            editor.add_cell(cell_count, v[i][j+1], v[i+1][j], v[i+1][j+1])
            cell_count += 1

    # vertical part of domain
    # vertices
    for i in range(num_vertices_side):
        z = i * spacing - 0.5
        for j in range(num_vertices_side):
            x = j * spacing + 0.5
            if not (i == center_index and j <= center_index):
                p = Point(x, 0.5, z)
                editor.add_vertex(vertex_count, p)
                v[i][j] = vertex_count
                vertex_count += 1
            else:
                v[i][j] += center_index

    # cells
    for i in range(num_vertices_side - 1):
        for j in range(num_vertices_side - 1):
            editor.add_cell(cell_count, v[i][j], v[i][j+1], v[i+1][j])
            cell_count += 1

            editor.add_cell(cell_count, v[i][j+1], v[i+1][j], v[i+1][j+1])
            cell_count += 1

    editor.close()

    return mesh
