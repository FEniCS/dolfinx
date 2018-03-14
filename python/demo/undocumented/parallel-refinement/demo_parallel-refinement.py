# Copyright (C) 2013 Chris N. Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

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
