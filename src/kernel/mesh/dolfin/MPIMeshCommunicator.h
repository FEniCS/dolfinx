// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by: Magnus Vikstrøm, 2007.
//
// First added:  2007-05-30
// Last changed: 2007-12-02

#ifndef __MPI_MESH_COMMUNICATOR_H
#define __MPI_MESH_COMMUNICATOR_H

#include <dolfin/MeshFunction.h>

namespace dolfin
{
  class Mesh;
  
  /// The class facilitates the transfer of a mesh between processes using MPI
  
  class MPIMeshCommunicator
  {
  public:
    
    /// Constructor
    MPIMeshCommunicator();

    /// Destructor
    ~MPIMeshCommunicator();

    /// Broadcast mesh to all processes
    static void broadcast(const Mesh& mesh);

    /// Receive mesh
    static void receive(Mesh& mesh);
    
    /// Broadcast mesh function to all processes
    static void broadcast(const MeshFunction<unsigned int>& mesh_function);

    /// Receive mesh function
    static void receive(MeshFunction<unsigned int>& mesh_function);

  };
}

#endif
