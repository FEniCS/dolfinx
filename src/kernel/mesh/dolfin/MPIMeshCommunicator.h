// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-05-30
// Last changed: 

#ifndef __MPI_MESH_COMMUNICATOR_H
#define __MPI_MESH_COMMUNICATOR_H

#include <dolfin/MeshFunction.h>

namespace dolfin
{
  class Mesh;
//  class MeshFunction<unsigned int>;
  
  /// The class facilitates the transfer of a mesh between processes using MPI
  
  class MPIMeshCommunicator
  {
  public:
    
    /// Constructor
    MPIMeshCommunicator();

    /// Destructor
    ~MPIMeshCommunicator();

    /// Broadcast mesh to all processes
    void broadcast(const Mesh& mesh);

    /// Receive mesh
    void receive(Mesh& mesh);
    
    /// Broadcast MeshFunction to all processes
    void broadcast(const MeshFunction<unsigned int>& mesh_function);

    /// Receive MeshFunction
    void receive(MeshFunction<unsigned int>& mesh_function);

  };

}

#endif
