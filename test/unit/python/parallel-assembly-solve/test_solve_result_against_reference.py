#!/usr/bin/env py.test
"""This file solves a simple reaction-diffusion problem and compares
the norm of the solution vector with a known solution (obtained when
running in serial). It is used for validating mesh partitioning and
parallel assembly/solve."""

# Copyright (C) 2009-2014 Anders Logg
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
# Modified by Garth N. Wells, 2013

from __future__ import print_function
import pytest
import sys
from dolfin import *
from dolfin_utils.test import *
from six import print_

# Relative tolerance for regression test
tol = 1e-10

def compute_norm(mesh, degree):
    "Solve on given mesh file and degree of function space."

    # Create function space
    V = FunctionSpace(mesh, "Lagrange", degree)

    # Define variational problem
    v = TestFunction(V)
    u = TrialFunction(V)
    f = Expression("sin(x[0])", degree=degree)
    g = Expression("x[0]*x[1]", degree=degree)
    a = dot(grad(v), grad(u))*dx + v*u*dx
    L = v*f*dx - v*g*ds

    # Compute solution
    w = Function(V)
    solve(a == L, w)

    # Return norm of solution vector
    return w.vector().norm("l2")

def print_reference(results):
    "Print nicely formatted values for gluing into code as a reference"
    MPI.barrier(mpi_comm_world())
    if MPI.rank(mpi_comm_world()) == 0:
        print_("reference = {", end=' ')
        for (i, result) in enumerate(results):
            if i > 0:
                print_("             ", end=' ')
            print_("(\"%s\", %d): %.16g" % result, end=' ')
            if i < len(results) - 1:
                print(",")
            else:
                print("}")
    MPI.barrier(mpi_comm_world())

def check_results(results, reference, tol):
    "Compare results with reference"
    errors = []
    for (mesh_file, degree, norm) in results:
        if (mesh_file, degree) not in reference:
            errors.append((mesh_file, degree, None, None, None))
        else:
            ref = reference[(mesh_file, degree)]
            diff = abs(norm - ref) / abs(ref)
            if diff >= tol:
                errors.append((mesh_file, degree, norm, ref, diff))
    return errors

def print_errors(errors):
    MPI.barrier(mpi_comm_world())
    if MPI.rank(mpi_comm_world()) == 0:
        print("Checking results")
        print("----------------")
        for (mesh_file, degree, norm, ref, diff) in errors:
            print_("(%s, %d):\t" % (mesh_file, degree), end=' ')
            if diff is None:
                print("missing reference")
            else:
                print_("*** ERROR", end=' ')
                print("(norm = %.16g, reference = %.16g, relative diff = %.16g)" % (norm, ref, diff))
    MPI.barrier(mpi_comm_world())

@use_gc_barrier
def test_computed_norms_against_references():
    # Reference values for norm of solution vector
    reference = { ("16x16 unit square", 1): 9.547454087328376 ,
                  ("16x16 unit square", 2): 18.42366670418269 ,
                  ("16x16 unit square", 3): 27.29583104732836 ,
                  ("16x16 unit square", 4): 36.16867128121694 ,
                  ("4x4x4 unit cube", 1): 12.23389289626038 ,
                  ("4x4x4 unit cube", 2): 28.96491629163837 ,
                  ("4x4x4 unit cube", 3): 49.97350551329799 ,
                  ("4x4x4 unit cube", 4): 74.49938266409099 }

    # Mesh files and degrees to check
    meshes = [(UnitSquareMesh(16, 16), "16x16 unit square"),
              (UnitCubeMesh(4, 4, 4),  "4x4x4 unit cube")]
    degrees = [1, 2, 3, 4]

    # For MUMPS, increase estimated require memory increase. Typically
    # required for high order elements on small meshes in 3D
    if has_petsc():
        PETScOptions.set("mat_mumps_icntl_14", 40)

    # Iterate over test cases and collect results
    results = []
    for mesh in meshes:
        for degree in degrees:
            gc_barrier()
            norm = compute_norm(mesh[0], degree)
            results.append((mesh[1], degree, norm))

    # Change option back to default
    if has_petsc():
        PETScOptions.set("mat_mumps_icntl_14", 20)

    # Check results
    errors = check_results(results, reference, tol)

    # Print errors for debugging if they fail
    if errors:
        print_errors(errors)

    # Print results for use as reference
    if any(e[-1] is None for e in errors): # e[-1] is diff
        print_reference(results)

    # A passing test should have no errors
    assert len(errors) == 0 # See stdout for detailed norms and diffs.
