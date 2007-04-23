#!/usr/bin/env python
#
# Copyright (C) 2006 Anders Logg.
# Licensed under the GNU LGPL Version 2.1.
#
# Run benchmark for computing overhead

from os import system
from time import time

# Log file
logfile = "bench-overhead.log"

# Run benchmark for cG(1)
tic = time()
system("./dolfin-ode-reaction cg fixed-point 1e-6 1000 5.0 parameters-overhead.xml")
tcg = time() - tic

# Run benchmark for mcG(1)
tic = time()
system("./dolfin-ode-reaction mcg fixed-point 1e-6 1000 5.0 parameters-overhead.xml")
tmcg = time() - tic

# Save timing to log file
system("dolfin-config --version >> " + logfile)
system("date >> " + logfile)
file = open(logfile, "a")
file.write("CPU time cG(1):  %g s\n" % tcg)
file.write("CPU time mcG(1): %g s\n" % tmcg)
file.write("----------------------------------------\n")
file.close()
