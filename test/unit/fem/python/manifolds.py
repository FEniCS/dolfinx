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
#
# First added:  2012-12-06
# Last changed: 2012-12-06

# MER: The solving test should be moved into test/regression/..., the
# evaluatebasis part should be moved into test/unit/FiniteElement.py

import unittest
from dolfin import *
from itertools import izip

import numpy
# Subdomain to extract bottom boundary.
class BottomEdge(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 0.0)

class Rotation(object):
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
                               for (j,a) in enumerate(self.invmat[i,:])])+")"

    def rotate(self, mesh):
        """Rotate mesh through phi then theta."""

        mesh.coordinates()[:,:] = \
            numpy.dot(mesh.coordinates()[:,:], self.mat.T)

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
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
    g = Expression("sin(5*x[0])")
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    return u

def poisson_manifold():
    # Create mesh
    cubemesh = UnitCubeMesh(32,32,2)

    boundarymesh = BoundaryMesh(cubemesh)

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
    f = Expression(("10*exp(-(pow(x[0] - %.17f, 2) "
                    + " + pow(x[1] - %.17f, 2)"
                    + " + pow(x[2] - %.17f, 2)) / 0.02)")\
                       % tuple(rotation.rotate_point([0.5,0.5,0])))
    g = Expression("sin(5*%s)"%rotation.x(0))

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    return u

def rotate_2d_mesh(theta):
    """Unit square mesh in 2D rotated through theta about the x and z axes."""

    cubemesh = UnitCubeMesh(1,1,1)
    boundarymesh = BoundaryMesh(cubemesh)
    mesh = SubMesh(boundarymesh, BottomEdge())

    mesh.init_cell_orientations(Expression(("0.","0.","1.")))

    rotation = Rotation(theta, theta)
    rotation.rotate(mesh)

    return mesh

class ManifoldSolving(unittest.TestCase):

    def test_poisson2D_in_3D(self):
        """This test solves Poisson's equation on a unit square in 2D,
        and then on a unit square embedded in 3D and rotated pi/4
        radians about each of the z and x axes."""

        # Boundary mesh not working in parallel
        if MPI.num_processes() > 1:
            return

        u_2D = poisson_2d()
        u_manifold = poisson_manifold()

        self.assertAlmostEqual(u_2D.vector().norm("l2"),
                               u_manifold.vector().norm("l2"), 10)
        self.assertAlmostEqual(u_2D.vector().max(),
                               u_manifold.vector().max(), 10)
        self.assertAlmostEqual(u_2D.vector().min(),
                               u_manifold.vector().min(), 10)

class ManifoldBasisEvaluation(unittest.TestCase):

    def test_basis_evaluation_2D_in_3D(self):
        """This test checks that basis functions and their
        derivatives are unaffected by rotations."""

        # Boundary mesh not working in parallel
        if MPI.num_processes() > 1:
            return

        self.basemesh = rotate_2d_mesh(0.0)
        self.rotmesh  = rotate_2d_mesh(numpy.pi/4)

        self.rotation = Rotation(numpy.pi/4, numpy.pi/4)

        for i in range(4):
            self.basis_test("CG", i+1)
        for i in range(5):
            self.basis_test("DG", i)

        for i in range(4):
            self.basis_test("RT", i+1, piola=True)
        for i in range(4):
            self.basis_test("BDM", i+1, piola=True)
        for i in range(4):
            self.basis_test("N1curl", i+1, piola=True)
        self.basis_test("BDFM", 2, piola=True)

    def basis_test(self, family, degree, piola=False):

        # Boundary mesh not working in parallel
        if MPI.num_processes() > 1:
            return

        parameters["form_compiler"]["no-evaluate_basis_derivatives"] = False

        f_base = FunctionSpace(self.basemesh, family, degree)
        f_rot = FunctionSpace(self.rotmesh, family, degree)

        points = numpy.array([[1.0, 1.0, 0.0],
                              [0.5, 0.5, 0.0],
                              [0.3, 0.7, 0.0],
                              [0.4, 0.0, 0.0]])

        for cell_base, cell_rot in izip(cells(self.basemesh), cells(self.rotmesh)):

            values_base = numpy.zeros(f_base.element().value_dimension(0))
            derivs_base = numpy.zeros(f_base.element().value_dimension(0)*3)
            values_rot = numpy.zeros(f_rot.element().value_dimension(0))
            derivs_rot = numpy.zeros(f_rot.element().value_dimension(0)*3)

            for i in range(f_base.element().space_dimension()):
                for point in points:
                    f_base.element().evaluate_basis(i, values_base,
                                                    point, cell_base)

                    f_base.element().evaluate_basis_derivatives(i, 1, derivs_base,
                                                                point, cell_base)

                    f_rot.element().evaluate_basis(i, values_rot,
                                                   self.rotation.rotate_point(point),
                                                   cell_rot)

                    f_base.element().evaluate_basis_derivatives(i, 1, derivs_rot,
                                                                self.rotation.rotate_point(point),
                                                                cell_rot)

                    if piola:
                        values_cmp = self.rotation.rotate_point(values_base)

                        derivs_rot2 = derivs_rot.reshape(f_rot.element().value_dimension(0),3)
                        derivs_base2 = derivs_base.reshape(f_base.element().value_dimension(0),3)
                        # If D is the unrotated derivative tensor, then RDR^T is the rotated version.
                        derivs_cmp = numpy.dot(self.rotation.mat, self.rotation.rotate_point(derivs_base2))
                    else:
                        values_cmp = values_base
                        # Rotate the derivative for comparison.
                        derivs_cmp = self.rotation.rotate_point(derivs_base)
                        derivs_rot2 = derivs_rot

                    self.assertAlmostEqual(abs(derivs_rot2-derivs_cmp).max(), 0.0, 10)
                    self.assertAlmostEqual(abs(values_cmp-values_rot).max(), 0.0, 10)

if __name__ == "__main__":
    print ""
    print "Testing solving and evaluate basis over manifolds"
    print "-------------------------------------------------"

    unittest.main()
