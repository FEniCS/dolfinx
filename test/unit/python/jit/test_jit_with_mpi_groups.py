#!/usr/bin/env py.test
"""Unit tests for the JIT compiler"""

# Copyright (C) 2014 Johan Hake
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

import pytest

# Cannot use decorators here as we would then need to import dolfin,
# and we need to import mpi and petsc4py before that.
def test_mpi_dependent_jiting():
    # FIXME: Not a proper unit test...
    try:
        import mpi4py.MPI as mpi
    except:
        return

    try:
        import petsc4py
    except:
        return
    
    # Set communicator and get process information
    comm = mpi.COMM_WORLD
    group = comm.Get_group()
    size = comm.Get_size()

    # Only consider parallel runs
    if size == 1:
        return

    rank = comm.Get_rank()
    group_0 = comm.Create(group.Incl(range(1)))
    group_1 = comm.Create(group.Incl(range(1,2)))

    if size > 2:
        group_2 = comm.Create(group.Incl(range(2,size)))

    # Init PETSc with the different groups
    if rank == 0:
        petsc4py.init(comm=group_0)
        import petsc4py.PETSc as petsc
        group_comm_0 = petsc.Comm(group_0)

    elif rank == 1:
        petsc4py.init(comm=group_1)
        import petsc4py.PETSc as petsc
        group_comm_1 = petsc.Comm(group_1)

    else:
        petsc4py.init(comm=group_2)
        import petsc4py.PETSc as petsc
        group_comm_2 = petsc.Comm(group_2)
        
    from dolfin import Expression, UnitSquareMesh, Function, TestFunction, \
         Form, FunctionSpace, dx, CompiledSubDomain
    
    if rank == 0:
        e = Expression("4", mpi_comm=group_comm_0)

    elif rank == 1:
        e = Expression("5", mpi_comm=group_comm_1)
        domain = CompiledSubDomain("on_boundary", mpi_comm=group_comm_1)

    else:
        mesh = UnitSquareMesh(group_comm_2, 2, 2)
        V = FunctionSpace(mesh, "P", 1)
        u = Function(V)
        v = TestFunction(V)
        Form(u*v*dx)
