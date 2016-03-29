#!/usr/bin/env py.test

"""Unit tests for FunctionSpace with constrained domain"""

# Copyright (C) 2012-2014 Garth N. Wells
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
# Modified by Mikael Mortensen 2013

from __future__ import print_function
import pytest
import numpy
from dolfin import *


class PeriodicBoundary2(SubDomain):
    def __init__(self, tolerance=DOLFIN_EPS):
        SubDomain.__init__(self, tolerance)
        self.tol = tolerance

    def inside(self, x, on_boundary):
        return bool(x[0] < self.tol and x[0] > -self.tol and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]


class PeriodicBoundary3(SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]
        y[2] = x[2]


def test_instantiation():
    """ A rudimentary test for instantiation"""

    sub_domain = PeriodicBoundary3()
    mesh = UnitCubeMesh(8, 8, 8)
    V = FunctionSpace(mesh, "CG", 1, constrained_domain=sub_domain)


def test_instantiation_mixed_element():
    """A rudimentary test for instantiation with mixed elements"""

    pbc = PeriodicBoundary2()
    mesh = UnitSquareMesh(8, 8)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    VV = FunctionSpace(mesh, P1*P1, constrained_domain=pbc)


def test_instantiation_mixed_element_real():
    """A rudimentary test for instantiation with mixed elements that
    include a real space
    """

    pbc = PeriodicBoundary2()
    mesh = UnitSquareMesh(8, 8)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    R = FiniteElement("Real", mesh.ufl_cell(), 0)
    VV = FunctionSpace(mesh, P1*R, constrained_domain=pbc)
    VV = FunctionSpace(mesh, R*P1, constrained_domain=pbc)


def test_instantiation_no_vertex_element_2D():
    """ A rudimentary test for instantiation for element that does
    not require number of vertices (2D)"""

    pbc = PeriodicBoundary2()
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "BDM", 1, constrained_domain=pbc)


def test_instantiation_no_vertex_element_3D():
    """ A rudimentary test for instantiation for element that does
    not require number of vertices (3D)"""

    pbc = PeriodicBoundary3()
    mesh = UnitCubeMesh(8, 8, 9)
    V = FunctionSpace(mesh, "BDM", 1, constrained_domain=pbc)


def test_director_lifetime():
    """Test for problems with objects with directors going out
    of scope"""

    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "Lagrange", 1,
                      constrained_domain=PeriodicBoundary2())


def test_tolerance():
    """Test tolerance for matching periodic mesh entities"""
    shift = 0.0001
    mesh = UnitSquareMesh(8, 8)

    # Randomly perturb mesh vertex coordinates
    mesh_perturb = Mesh(mesh)
    import random
    for x in mesh_perturb.coordinates():
        x[0] += random.uniform(-shift, shift)
        x[1] += random.uniform(-shift, shift)

    pbc = PeriodicBoundary2()
    pbc_tol = PeriodicBoundary2(2*shift)

    for dim in range(mesh.geometry().dim()):
        periodic_pairs = PeriodicBoundaryComputation.compute_periodic_pairs(mesh, pbc, dim)
        num_periodic_pairs0 = len(periodic_pairs)

        periodic_pairs = PeriodicBoundaryComputation.compute_periodic_pairs(mesh_perturb,
                                                                            pbc_tol, dim)
        num_periodic_pairs1 = len(periodic_pairs)
        assert num_periodic_pairs0 == num_periodic_pairs1


def test_solution():
    """Test periodic constrained domain by checking solution to a PDE."""

    # Create mesh and constrained FunctionSpace
    mesh = UnitSquareMesh(8, 8)
    pbc = PeriodicBoundary2()
    V = FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)

    class DirichletBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return bool((x[1] < DOLFIN_EPS or x[1] > (1.0 - DOLFIN_EPS)) and
                        on_boundary)

    # Dirichlet boundary condition
    dirichlet_boundary = DirichletBoundary()
    bc0 = DirichletBC(V, 0.0, dirichlet_boundary)
    bcs = [bc0]

    # Define variational problem, linear formulation
    u, v = TrialFunction(V), TestFunction(V)
    f = Expression("sin(x[0])", degree=2)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bcs)

    assert round(u.vector().norm("l2") - 0.3368694028630991, 10) == 0

    # Define variational problem, nonlinear formulation
    u, v = Function(V), TestFunction(V)
    f = Expression("sin(x[0])", degree=2)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    F = a - L

    # Compute solution
    solve(F == 0, u, bcs)

    assert round(u.vector().norm("l2") - 0.3368694028630991, 10) == 0
