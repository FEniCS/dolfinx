// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-30
// Last changed:

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/MeshFunction.h>
#include <dolfin/MPIMeshCommunicator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MPIMeshCommunicator::MPIMeshCommunicator()
{
  MPI_Init(0, 0);
}
//-----------------------------------------------------------------------------
MPIMeshCommunicator::~MPIMeshCommunicator()
{
  MPI_Finalize();
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::broadcast(const Mesh& mesh)
{
  int process_int;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_int);
  unsigned int this_process = process_int;

  int broadcast_message = -1;

  if(this_process == 0)
  {
    broadcast_message = 42;
  }
 
  MPI_Bcast(&broadcast_message, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  int sum = this_process + broadcast_message;
  cout << "Finished mesh broadcast on process " << this_process << " sum: " << sum << endl;
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::receive(Mesh& mesh)
{
  error("MPI receival of meshes not yet implemented");
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::broadcast(const MeshFunction<unsigned int>& mesh_function)
{
  int process_int;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_int);
  unsigned int this_process = process_int;

  int broadcast_message = -1;

  if(this_process == 0)
  {
    broadcast_message = 42;
  }
 
  MPI_Bcast(&broadcast_message, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  int sum = this_process + broadcast_message;
  cout << "Finished meshfunction broadcast on process" << this_process << " sum: " << sum << endl;
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::receive(MeshFunction<unsigned int>& mesh_function)
{
  error("MPI receival of a MeshFunction not yet implemented");
}
//-----------------------------------------------------------------------------

