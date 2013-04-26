# Copyright (C) 2013 Chris N. Richardson
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
# First added:  2013-04-26
# Last changed: 2013-04-26

from dolfin import *
import os.path

if(MPI.num_processes() == 1):
    print "This demo is intended to be run in parallel (using mpirun)"
    print "However, it will also run on a single process"

# make a Function F=x*y on a square mesh
mesh = UnitSquareMesh(20,20)
Q = FunctionSpace(mesh, "CG", 1)
F = Function(Q)
E = Expression("x[0]*x[1]")
F.interpolate(E)

# output in XDMF format for visualisation
# view it in paraview or visit
F_file = File("function.xdmf")
F_file << F

file_exists = os.path.exists("mesh.xdmf")
MPI.barrier() # prevents race condition

# Check for file "mesh.xdmf" in folder, and read in
if file_exists:
    mesh = Mesh("mesh.xdmf")
    if MPI.process_number() == 0:
        print "Read mesh using ", MPI.num_processes(), " processes."
    print mesh
else:
    # mesh.xdmf does not exist, so create
    M_file = File("mesh.xdmf")
    M_file << mesh
    if MPI.process_number() == 0:
        print "Wrote mesh to file using ", MPI.num_processes(), " processes."
        print "Try rerunning the demo with a different number of processes."
    
