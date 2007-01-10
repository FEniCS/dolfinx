#!/usr/bin/env python
#
# Copyright (C) 2006 Garth N. Wells.
# Licensed under the GNU GPL Version 2.
#
# Run benchmarks for linear algebra.

from os import system
from time import time

# Log file
logfile = "bench.log"

tic = time()

# Run benchmark for vectors
system("./vector/dolfin-vector | tee bench.tmp")

# Run benchmark for spase matrices
system("./sparse-matrix/dolfin-sparse-matrix | tee -a bench.tmp")

toc = time() - tic

# Save results to log file
system("pkg-config --modversion dolfin >> " + logfile)
system("date >> " + logfile)
file = open(logfile, "a")
tempfile = open('./bench.tmp', 'r')
file.write(tempfile.read())
file.write("Total CPU time: %g\n" % toc)
file.write("----------------------------------------\n")
tempfile.close()
file.close()

# Print finished message
print "Finished linear algebra benchmarks. See '"+logfile+"' for the results."
