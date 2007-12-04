// Copyright (C) 2007 Magnus Vikstrøm.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
// Modified by Anders Logg, 2007.
//
// First added:  2007-11-30
// Last changed: 2007-12-04

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/MeshFunction.h>
#include <dolfin/MPIManager.h>

// Initialize static data
dolfin::MPIManager dolfin::MPIManager::mpi;

using namespace dolfin;

#ifdef HAVE_MPI_H

//-----------------------------------------------------------------------------
MPIManager::MPIManager()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MPIManager::~MPIManager()
{
  dolfin_debug("Calling MPIManager::finalize() in destructor of singleton MPIManager instance");
  MPIManager::finalize();
}
//-----------------------------------------------------------------------------
void MPIManager::init()
{
  int initialized;
  MPI_Initialized(&initialized);
  if (initialized)
    return;

  dolfin_debug("Initializing MPI");
  MPI_Init(0, 0);
}
//-----------------------------------------------------------------------------
void MPIManager::finalize()
{
  int initialized;
  MPI_Initialized(&initialized);
  if (initialized)
  {
    dolfin_debug("Finalizing MPI");
    MPI_Finalize();
  }
}
//-----------------------------------------------------------------------------
dolfin::uint MPIManager::processNumber()
{
  MPIManager::init();

  int this_process;
  MPI_Comm_rank(MPI_COMM_WORLD, &this_process);

  dolfin_debug1("MPIManager: Process number is %d", this_process);

  return static_cast<uint>(this_process);
}
//-----------------------------------------------------------------------------
dolfin::uint MPIManager::numProcesses()
{
  MPIManager::init();

  int num_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  dolfin_debug1("MPIManager: Number of processes is %d", num_processes);

  return static_cast<uint>(num_processes);
}
//-----------------------------------------------------------------------------
bool MPIManager::broadcast()
{
  // Always broadcast from processor number 0
  return numProcesses() > 0 && processNumber() == 0;
}
//-----------------------------------------------------------------------------
bool MPIManager::receive()
{
  // Always receive on processors with numbers > 0
  return numProcesses() > 0 && processNumber() > 0;
}
//-----------------------------------------------------------------------------

#else

//-----------------------------------------------------------------------------
MPIManager::MPIManager()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MPIManager::~MPIManager()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MPIManager::init()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MPIManager::finalize()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint MPIManager::processNumber()
{
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint MPIManager::numProcesses()
{
  return 1;
}
//-----------------------------------------------------------------------------
bool MPIManager::broadcast();
{
  return false;
}
//-----------------------------------------------------------------------------
bool MPIManager::receive()
{
  return false;
}
//-----------------------------------------------------------------------------

#endif
