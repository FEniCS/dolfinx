// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by: Magnus Vikstrøm, 2007.
//
// First added:  2007-05-30
// Last changed: 2007-12-01

#ifndef __MPI_MESH_COMMUNICATOR_H
#define __MPI_MESH_COMMUNICATOR_H

#include <dolfin/MeshFunction.h>

#ifdef HAVE_MPI_H
  #include <mpi.h>
#endif

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
    static void broadcast(const Mesh& mesh);

    /// Receive mesh
    static void receive(Mesh& mesh);
    
    /// Broadcast MeshFunction to all processes
    static void broadcast(const MeshFunction<unsigned int>& mesh_function);

    /// Receive MeshFunction
    static void receive(MeshFunction<unsigned int>& mesh_function);

    /// Return proccess number
    static int processNum();
  };
}

#endif
