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

# Run benchmark for cG(1)
tic = time()
system("./dolfin-ode-wave cg")
tcg = time() - tic

# Run benchmark for mcG(1)
tic = time()
system("./dolfin-ode-wave mcg")
tmcg = time() - tic

# Save timing to log file
system("pkg-config --modversion dolfin >> " + logfile)
system("date +'%Y-%m-%d %H:%M:%S' >> " + logfile)
system("uname -snm >> " + logfile)
file = open(logfile, "a")
file.write("\n");
file.write("CPU time cG(1):  %g s\n" % tcg)
file.write("CPU time mcG(1): %g s\n" % tmcg)
file.write("----------------------------------------\n")
file.close()
