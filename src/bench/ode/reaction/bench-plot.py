#!/usr/bin/env python
#
# Copyright (C) 2006 Anders Logg.
# Licensed under the GNU GPL Version 2.
#
# Run benchmark to generate data for plots

from os import system
from time import time

system("./dolfin-ode-reaction mcg fixed-point 1e-6 1000 5.0 parameters-plot.xml")
system("mv primal.m solution.m")

system("./dolfin-ode-reaction mcg fixed-point 5e-8 1000 5.0 parameters-plot.xml")
system("mv primal.m solution_fine_tolerance.m")

system("./dolfin-ode-reaction mcg fixed-point 1e-6 16000 80.0 parameters-plot.xml")
system("mv primal.m solution_large_domain.m")
