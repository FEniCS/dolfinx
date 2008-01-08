// Copyright (C) 2007 Magnus Vikstrøm.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007, 2008.
// Modified by Anders Logg, 2007.
//
// First added:  2007-11-30
// Last changed: 2008-01-07

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/MeshFunction.h>
#include <dolfin/MPI.h>
#include <dolfin/SubSystemsManager.h>

#ifdef HAVE_MPI_H
  #include <mpi.h>
#endif

//-----------------------------------------------------------------------------
#ifdef HAVE_MPI_H
dolfin::uint dolfin::MPI::processNumber()
{
  SubSystemsManager::initMPI();

  int this_process;
  MPI_Comm_rank(MPI_COMM_WORLD, &this_process);

  dolfin_debug1("MPI: Process number is %d", this_process);

  return static_cast<uint>(this_process);
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::numProcesses()
{
  SubSystemsManager::initMPI();

  int num_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  dolfin_debug1("MPI: Number of processes is %d", num_processes);

  return static_cast<uint>(num_processes);
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::broadcast()
{
  // Always broadcast from processor number 0
  return numProcesses() > 1 && processNumber() == 0;
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::receive()
{
  // Always receive on processors with numbers > 0
  return numProcesses() > 1 && processNumber() > 0;
}
//-----------------------------------------------------------------------------

#else

//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::processNumber()
{
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::MPI::numProcesses()
{
  return 1;
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::broadcast()
{
  return false;
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::receive()
{
  return false;
}
//-----------------------------------------------------------------------------

#endif
