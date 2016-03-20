"This demo program demonstrates how to include additional C++ code in DOLFIN."

# Copyright (C) 2013 Kent-Andre Mardal, Mikael Mortensen, Johan Hake
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2013-04-02

from __future__ import print_function
from dolfin import *
import numpy
import os

header_file = open("Probe/Probe.h")
code = "\n".join(header_file.readlines())
compiled_module = compile_extension_module(code=code, source_directory="Probe", \
                                           sources=["Probe.cpp"], \
                                           include_dirs=[".", os.path.abspath("Probe")])

mesh = UnitCubeMesh(10, 10, 10)
V = FunctionSpace(mesh, 'CG', 1)

x = numpy.array((0.5, 0.5, 0.5))
probe = compiled_module.Probe(x, V)

# Just create some random data to be used for probing
u0 = interpolate(Expression('x[0]', degree=1), V)
probe.eval(u0)
print("number of probes: ", probe.value_size())
print("value at first probe: ", probe.get_probe(0))
