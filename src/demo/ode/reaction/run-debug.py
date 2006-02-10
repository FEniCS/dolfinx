#!/usr/bin/python
#
# Copyright (C) 2005 Anders Logg.
# Licensed under the GNU GPL Version 2.
#
# Run benchmarks for test problem and collect results

from os import system
from commands import getoutput
from checkerror import checkerror

def run_set(method, solver, tolmax, k0, kmax, T, gamma, N):
    "Run set of tests for given data."

    # Open file for storing benchmark results
    file = open("bench.log", "a")
    # Open file for storing output
    file2 = open("output.log", "w")
    
    # Write results header
    file.write("\n%s/%s gamma = %.3e\n" % (method, solver, gamma))
    file.write("---------------------------------\n")
    file.write("\nTOL\tError\t\tCPU time\tSteps\tIterations\tIndex\n")
    file.write("--------------------------------------------------------------------------------------\n")
    
    for tol in [1e-5]:
    
        # Run program
        print "Running %s/%s at tolerance %.1e" % (method, solver, tol)
        output = \
            getoutput("./dolfin-ode-reaction %s %s %e %e %e %e %e %d %s" % \
                (method, solver, tol, k0, kmax, T, gamma, N, \
                 "debug-parameters.xml"))

        file2.write(output)

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
        error = checkerror("solution.data", "reference-gamma-1000-N-20.data")

        # Write results to file
        file.write("%.1e\t%.3e\t%s\t\t%s (%s)\t%s (%s)\t%s\n" % \
                   (tol, error, cputime, steps, rejected, global_iter, local_iter, index))

    # Close file
    file.close()
    file2.close()

# Move old bench.log to bench.bak
system("mv bench.log bench.bak")


# Run sets of benchmarks for gamma = 100
#run_set("cg",  "newton",      1e-3, 1e-2, 1e-2, 3.0, 100.0, 1000)
#run_set("cg",  "fixed-point", 1e-3, 1e-2, 1e-2, 3.0, 100.0, 1000)
#run_set("mcg", "newton",      1e-3, 1e-2, 1e-2, 3.0, 100.0, 1000)
#run_set("mcg", "fixed-point", 1e-3, 1e-2, 1e-2, 3.0, 100.0, 1000)

# Run sets of benchmarks for gamma = 1000
#run_set("cg",  "newton",      1e-6, 1e-5, 1e-3, 1.0, 1000.0, 1000)
#run_set("cg",  "fixed-point", 1e-6, 1e-5, 1e-3, 1.0, 1000.0, 20)
#run_set("mcg", "newton",      1e-6, 1e-5, 1e-3, 1.0, 1000.0, 1000)
#run_set("mcg", "fixed-point", 1e-6, 1e-5, 1e-3, 1.0, 1000.0, 20)

# Debug run
run_set("cg",  "fixed-point", 1e-6, 1e-4, 1e-3, 0.1, 1000.0, 20)
run_set("mcg", "fixed-point", 1e-6, 1e-4, 1e-3, 0.1, 1000.0, 20)

#run_set("cg",  "fixed-point", 1e-6, 1e-8, 1e-4, 0.1, 1000.0, 20)

# Print results
system("cat bench.log")
