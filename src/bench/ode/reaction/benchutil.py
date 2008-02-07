#!/usr/bin/python
#
# Copyright (C) 2005-2006 Anders Logg.
# Licensed under the GNU LGPL Version 2.1.
#
# Utilities for running benchmark

from scipy.io.array_import import read_array
from commands import getoutput
from shutil import copy, move

def check_error(solution, reference):
    U = read_array(solution)
    u = read_array(reference)
    e = max(abs(U - u))
    return e

def run_set(method, solver, tols, sizes, logfile):
    "Run set of tests for given data."

    # Open file for storing benchmark results
    file = open(logfile, "a")
    
    # Write results header
    file.write("\n%s/%s\n" % (method, solver))
    file.write("--------------\n")
    file.write("\ntol\tN\tError\t\tCPU time\tSteps\tIterations\tIndex\n")
    file.write("---------------------------------------------------------------------------------------\n")

    for N in sizes:
        for tol in tols:

            # Length of domain
            L = 5.0 * N / 1000.0

            # Run program
            print "Running benchmark problem for %s/%s at tol = %.1e N = %d" % (method, solver, tol, N)
            output = getoutput("./dolfin-ode-reaction %s %s %e %d %e parameters-bench.xml" % \
                               (method, solver, tol, N, L))

            print output

            # Check if we got any solution
            if len(output.split("Solution stopped")) > 1:
                file.write("Unable to compute solution\n")
                continue

            # Parse output
            cputime = output.split("computed in ")[1].split(" ")[0]
            global_iter = output.split("iterations per step: ")[1].split("\n")[0]
            local_iter = output.split("per global iteration: ")[1].split("\n")[0]
            steps = output.split("(macro) time steps: ")[1].split("\n")[0]
            rejected = output.split("rejected time steps: ")[1].split("\n")[0]
            if method == "mcg":
                index = output.split("efficiency index: ")[1].split("\n")[0]
            else:
                index = 1.0

            # Compute error
            error = check_error("solution.data", "reference_" + str(N) + ".data")
            
            # Write results to file
            file.write("%.1e\t%d\t%.3e\t%s\t\t%s (%s)\t%s (%s)\t%s\n" % \
                       (tol, N, error, cputime, steps, rejected, global_iter, local_iter, index))
            file.flush()
                
    # Close file
    file.close()
