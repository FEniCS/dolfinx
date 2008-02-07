#!/usr/bin/env python
#
# Copyright (C) 2005-2008 Anders Logg.
# Licensed under the GNU LGPL Version 2.1.
#
# Run benchmarks for varying tolerance

from os import system
from shutil import copy, move
from benchutil import *

# Log file
logfile = "bench-tol.log"

# Parameter set for benchmarks
tols  = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]
sizes = [1000]

# Write version and date to log file
system("pkg-config --modversion dolfin >> " + logfile)
system("date +'%Y-%m-%d %H:%M:%S' >> " + logfile)
system("uname -snm >> " + logfile)
file = open(logfile, "a")
file.write("\n");

# Run sets of benchmarks
run_set("cg",  "fixed-point", tols, sizes, logfile)
run_set("mcg", "fixed-point", tols, sizes, logfile)
#run_set("cg",  "newton",      tols, sizes, logfile)
#run_set("mcg", "newton",      tols, sizes, logfile)

file.write("\n");
file.write("---------------------------------------------------------------------------------------\n")
