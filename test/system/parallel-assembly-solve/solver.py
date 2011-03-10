"""This file solves a simple reaction-diffusion problem and compares
the norm of the solution vector with a known solution (obtained when
running in serial). It is used for validating mesh partitioning and
parallel assembly/solve."""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2009-08-17 -- 2009-08-17"
__copyright__ = "Copyright (C) 2009 Anders Logg"
__license__  = "GNU LGPL version 2.1"

import sys
from dolfin import *

# Relative tolerance for regression test
tol = 1e-10

def solve(mesh_file, degree):
    "Solve on given mesh file and degree of function space."

    # Create mesh and define function space
    mesh = Mesh(mesh_file);
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
#reference = { ("unitsquare.xml.gz", 1): 7.821707395007537 ,
#              ("unitsquare.xml.gz", 2): 15.18829494599347 ,
#              ("unitsquare.xml.gz", 3): 22.55234140275229 ,
#              ("unitsquare.xml.gz", 4): 29.91638783448794 ,
#              ("unitsquare.xml.gz", 5): 37.28043428001642 ,
#              ("unitcube.xml.gz", 1): 3.647913575216382 ,
#              ("unitcube.xml.gz", 2): 8.523874310611367 ,
#              ("unitcube.xml.gz", 3): 14.55432230797502 ,
#              ("unitcube.xml.gz", 4): 21.57286638104142 ,
#              ("unitcube.xml.gz", 5): 29.45598181177814 }
reference = { ("unitsquare.xml.gz", 1): 9.547454087327344 ,
              ("unitsquare.xml.gz", 2): 18.42366670418527 ,
              ("unitsquare.xml.gz", 3): 27.29583104741712 ,
              ("unitsquare.xml.gz", 4): 36.1686712809094 ,
              ("unitcube.xml.gz", 1): 8.876490653853809 ,
              ("unitcube.xml.gz", 2): 19.99081167299566 ,
              ("unitcube.xml.gz", 3): 33.85477561286852 ,
              ("unitcube.xml.gz", 4): 49.97357666762962 }

# Mesh files and degrees to check
mesh_files = ["unitsquare.xml.gz", "unitcube.xml.gz"]
degrees = [1, 2, 3, 4]

## Iterate over test cases and collect results
results = []
for mesh_file in mesh_files:
    for degree in degrees:
        norm = solve(mesh_file, degree)
        results.append((mesh_file, degree, norm))

# Uncomment to print results for use as reference
#print_reference(results)

# Check results
status = check_results(results, reference, tol)

# Resturn exit status
sys.exit(status)
