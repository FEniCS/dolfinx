// Copyright (C) 2007 Magnus Vikstrøm.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
//
// First added:  2007-11-30
// Last changed: 2007-12-02

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/MeshFunction.h>
#include <dolfin/MPIManager.h>

using namespace dolfin;

#ifdef HAVE_MPI_H
//-----------------------------------------------------------------------------
MPIManager::MPIManager()
{
  //Do nothing
}
//-----------------------------------------------------------------------------
MPIManager::~MPIManager()
{
  MPIManager::finalize();
}
//-----------------------------------------------------------------------------
void MPIManager::finalize()
{
  int inited;
  MPI_Initialized(&inited);
  if (inited)
  {
    dolfin_debug("Finalizing MPI");
    MPI_Finalize();
  }
}
//-----------------------------------------------------------------------------
void MPIManager::init()
{
  int inited;
  MPI_Initialized(&inited);
  if (inited)
  {
    return;
  }
  dolfin_debug("Initing MPI");
  MPI_Init(0, 0);
}
//-----------------------------------------------------------------------------
int MPIManager::processNum()
{
  dolfin_debug("MPIManagor::processNum");
  MPIManager::init();

  int this_process;
  MPI_Comm_rank(MPI_COMM_WORLD, &this_process);

  return this_process;
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
int MPIManager::processNum()
{
  return 0;
}
//-----------------------------------------------------------------------------
void MPIManager::finalize()
{
  // Do nothing
}
//-----------------------------------------------------------------------------

#endif
