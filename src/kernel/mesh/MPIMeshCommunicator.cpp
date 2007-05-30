// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
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
  // Do nothing
}
//-----------------------------------------------------------------------------
MPIMeshCommunicator::~MPIMeshCommunicator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::broadcast(const Mesh& mesh)
{
  error("MPI broadcast of meshes not yet implemented");
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::receive(Mesh& mesh)
{
  error("MPI receival of meshes not yet implemented");
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::broadcast(const MeshFunction<unsigned int>& mesh_function)
{
  error("MPI broadcast of a MeshFunction not yet implemented");
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::receive(MeshFunction<unsigned int>& mesh_function)
{
  error("MPI receival of a MeshFunction not yet implemented");
}
//-----------------------------------------------------------------------------

