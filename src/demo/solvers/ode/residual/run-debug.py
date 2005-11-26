#!/usr/bin/python
#
# Copyright (C) 2005 Anders Logg.
# Licensed under the GNU GPL Version 2.
#
# Run benchmarks for test problem and collect results

from os import system
from commands import getoutput

outputcg = getoutput("./dolfin-ode-test cg")
outputmcg = getoutput("./dolfin-ode-test mcg")

filecg = open("outputcg.log", "w")
filecg.write(outputcg)
filecg.close()

filemcg = open("outputmcg.log", "w")
filemcg.write(outputmcg)
filemcg.close()
