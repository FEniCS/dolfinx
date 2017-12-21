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

# Create mesh
mesh = UnitSquareMesh(20, 20)

# Create MeshFunction to hold cell process rank
processes = MeshFunction('size_t', mesh, mesh.topology().dim(), MPI.rank(mesh.mpi_comm()))

# Output cell distribution to VTK file
file = File("processes.pvd")
file << processes

# Mark all cells on process 0 for refinement
marker = MeshFunction('bool', mesh, mesh.topology().dim(), (MPI.rank(mesh.mpi_comm()) == 0))

# Refine mesh, but keep all news cells on parent process
mesh0 = refine(mesh, marker, False)

# Create MeshFunction to hold cell process rank for refined mesh
processes1 = MeshFunction('size_t', mesh0, mesh0.topology().dim(), MPI.rank(mesh.mpi_comm()))
file << processes1

# Refine mesh, but this time repartition the mesh after refinement
mesh1 = refine(mesh, marker, True)

# Create MeshFunction to hold cell process rank for refined mesh
processes2 = MeshFunction('size_t', mesh1, mesh1.topology().dim(), MPI.rank(mesh.mpi_comm()))
file << processes2
