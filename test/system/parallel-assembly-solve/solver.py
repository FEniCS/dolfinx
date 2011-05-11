"""This file solves a simple reaction-diffusion problem and compares
the norm of the solution vector with a known solution (obtained when
running in serial). It is used for validating mesh partitioning and
parallel assembly/solve."""

# Copyright (C) 2009 Anders Logg
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2009-08-17
# Last changed: 2009-08-17

import sys
from dolfin import *

# Relative tolerance for regression test
tol = 1e-10

def solve(mesh, degree):
    "Solve on given mesh file and degree of function space."

    # Create function space
    V = FunctionSpace(mesh, "CG", degree)

    # Define variational problem
    v = TestFunction(V)
    u = TrialFunction(V)
    f = Expression("sin(x[0])", degree=degree)
    g = Expression("x[0]*x[1]", degree=degree)
    a = dot(grad(v), grad(u))*dx + v*u*dx
    L = v*f*dx - v*g*ds

    # Compute solution
    print "Degree:", degree
    problem = VariationalProblem(a, L)
    #problem.parameters["linear_solver"] = "iterative"
    #problem.parameters["krylov_solver"]["relative_tolerance"] = 1e-15
    u = problem.solve()

    # Return norm of solution vector
    return u.vector().norm("l2")

def print_reference(results):
    "Print nicely formatted values for gluing into code as a reference"
    print "reference = {",
    for (i, result) in enumerate(results):
        if i > 0:
            print "             ",
        print "(\"%s\", %d): %.16g" % result,
        if i < len(results) - 1:
            print ","
        else:
            print "}"

def check_results(results, reference, tol):
    "Compare results with reference"

    status = True

    if not MPI.process_number() == 0:
        return

    print "Checking results"
    print "----------------"

    for (mesh_file, degree, norm) in results:
        print "(%s, %d):\t" % (mesh_file, degree),
        if (mesh_file, degree) in reference:
            ref = reference[(mesh_file, degree)]
            diff =  abs(norm - ref) / abs(ref)
            if diff < tol:
                print "OK",
            else:
                status = False
                print "*** ERROR",
            print "(norm = %.16g, reference = %.16g, relative diff = %.16g)" % (norm, ref, diff)
        else:
            print "missing reference"

    return status

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
meshes= [(UnitSquare(16, 16), "16x16 unit square"),\
         (UnitCube(4, 4, 4),  "4x4x4 unit cube")]
degrees = [1, 2, 3, 4]

## Iterate over test cases and collect results
results = []
for mesh in meshes:
    for degree in degrees:
        norm = solve(mesh[0], degree)
        results.append((mesh[1], degree, norm))

# Uncomment to print results for use as reference
#print_reference(results)

# Check results
status = check_results(results, reference, tol)

# Resturn exit status
sys.exit(status)
