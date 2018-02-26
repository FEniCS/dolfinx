"""This file solves a simple reaction-diffusion problem and compares
the norm of the solution vector with a known solution (obtained when
running in serial). It is used for validating mesh partitioning and
parallel assembly/solve."""

# Copyright (C) 2009-2014 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import sys
from dolfin import *
from dolfin_utils.test import *

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
    MPI.barrier(MPI.comm_world)
    if MPI.rank(MPI.comm_world) == 0:
        print("reference = {", end=' ')
        for (i, result) in enumerate(results):
            if i > 0:
                print("             ", end=' ')
            print("(\"%s\", %d): %.16g" % result, end=' ')
            if i < len(results) - 1:
                print(",")
            else:
                print("}")
    MPI.barrier(MPI.comm_world)

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
    MPI.barrier(MPI.comm_world)
    if MPI.rank(MPI.comm_world) == 0:
        print("Checking results")
        print("----------------")
        for (mesh_file, degree, norm, ref, diff) in errors:
            print("(%s, %d):\t" % (mesh_file, degree), end=' ')
            if diff is None:
                print("missing reference")
            else:
                print("*** ERROR", end=' ')
                print("(norm = %.16g, reference = %.16g, relative diff = %.16g)" % (norm, ref, diff))
    MPI.barrier(MPI.comm_world)

def test_computed_norms_against_references():
    # Reference values for norm of solution vector
    reference = { ("16x16 unit tri square", 1): 9.547454087328376 ,
                  ("16x16 unit tri square", 2): 18.42366670418269 ,
                  ("16x16 unit tri square", 3): 27.29583104732836 ,
                  ("16x16 unit tri square", 4): 36.16867128121694 ,
                  ("4x4x4 unit tet cube", 1): 12.23389289626038 ,
                  ("4x4x4 unit tet cube", 2): 28.96491629163837 ,
                  ("4x4x4 unit tet cube", 3): 49.97350551329799 ,
                  ("4x4x4 unit tet cube", 4): 74.49938266409099 ,
                  ("16x16 unit quad square", 1): 9.550848071820747 ,
                  ("16x16 unit quad square", 2): 18.423668706176354 ,
                  ("16x16 unit quad square", 3): 27.295831017251672 ,
                  ("16x16 unit quad square", 4): 36.168671281610855 ,
                  ("4x4x4 unit hex cube", 1): 12.151954087339782 ,
                  ("4x4x4 unit hex cube", 2): 28.965646690046885 ,
                  ("4x4x4 unit hex cube", 3): 49.97349423895635 ,
                  ("4x4x4 unit hex cube", 4): 74.49938136593539 }

    # Mesh files and degrees to check
    meshes = [(UnitSquareMesh(16, 16), "16x16 unit tri square"),
              (UnitCubeMesh(4, 4, 4),  "4x4x4 unit tet cube"),
              (UnitSquareMesh.create(16, 16, CellType.Type.quadrilateral), "16x16 unit quad square"),
              (UnitCubeMesh.create(4, 4, 4, CellType.Type.hexahedron), "4x4x4 unit hex cube")]
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
