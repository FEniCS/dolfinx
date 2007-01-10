#!/usr/bin/env python
#
# Copyright (C) 2006 Anders Logg.
# Licensed under the GNU GPL Version 2.
#
# Run benchmark for ODE test problem.

from os import system
from time import time

# Log file
logfile = "bench.log"

# Run benchmark
tic = time()
system("./dolfin-ode-reaction mcg fixed-point 1e-3 2000 1.0 parameters-bench.xml")
toc = time() - tic

# Save timing to log file
system("pkg-config --modversion dolfin >> " + logfile)
system("date +'%Y-%m-%d %H:%M:%S' >> " + logfile)
system("uname -snm >> " + logfile)
file = open(logfile, "a")
file.write("\n");
file.write("CPU time: %g\n" % toc)
file.write("----------------------------------------\n")
file.close()
