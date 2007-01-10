#!/usr/bin/env python
#
# Copyright (C) 2006 Anders Logg.
# Licensed under the GNU GPL Version 2.
#
# Run benchmark for fem assemble and solve test problem.

from os import system
from time import time

# Log file
logfile = "bench.log"

# Run benchmark
tic = time()
system("./dolfin-fem-convergence")
toc = time() - tic

# Save timing to log file
system("pkg-config --modversion dolfin >> " + logfile)
system("date >> " + logfile)
file = open(logfile, "a")
file.write("CPU time: %g\n" % toc)
file.write("----------------------------------------\n")
file.close()
