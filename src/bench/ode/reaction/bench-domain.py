#!/usr/bin/env python
#
# Copyright (C) 2005-2006 Anders Logg.
# Licensed under the GNU GPL Version 2.
#
# Run benchmarks for varying size of domain

from os import system
from shutil import copy, move
from benchutil import *

# Log file
logfile = "bench-domain.log"

# Parameter set for benchmarks
tols  = [1e-6]
sizes = [1000, 2000, 4000, 8000, 16000]

# Write version and date to log file
system("dolfin-config --version >> " + logfile)
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
