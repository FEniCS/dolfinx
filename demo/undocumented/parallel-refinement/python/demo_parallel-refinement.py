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

if(MPI.num_processes() == 1):
    print "This demo is intended to be run in parallel (using mpirun)"
    print "However, it will also run on a single process"

mesh = UnitSquareMesh(20,20)

# Output cell distribution amongst processes
file = File("processes.pvd")
processes = MeshFunction('size_t', mesh, mesh.topology().dim(), MPI.process_number())
file << processes

# Mark all cells on process 0 for refinement
marker = MeshFunction('bool', mesh, mesh.topology().dim(), (MPI.process_number() == 0))

# Do refinement, but keep all new cells on parent process
parameters['mesh_partitioner'] = 'None'
mesh2 = refine(mesh, marker)
processes = MeshFunction('size_t', mesh2, mesh2.topology().dim(), MPI.process_number())
file << processes

# try to find a repartitioning partitioner, and do the previous refinement again
parameters['partitioning_approach'] = 'REPARTITION'
if has_parmetis():
    parameters['mesh_partitioner'] = 'ParMETIS'
elif has_trilinos():
    parameters['mesh_partitioner'] = 'Zoltan_PHG'
else:
    parameters['mesh_partitioner'] = 'SCOTCH'

mesh2 = refine(mesh, marker)
processes = MeshFunction('size_t', mesh2, mesh2.topology().dim(), MPI.process_number())
file << processes
