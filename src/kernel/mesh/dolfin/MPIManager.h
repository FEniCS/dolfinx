// Copyright (C) 2007 Magnus Vikstrøm.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-11-30
// Last changed: 2007-12-01

#ifndef __MPI_MANAGER_H
#define __MPI_MANAGER_H

#include <dolfin/MeshFunction.h>

#ifdef HAVE_MPI_H
  #include <mpi.h>
#endif

namespace dolfin
{
  class Mesh;
//  class MeshFunction<unsigned int>;
  
  /// The class facilitates the transfer of a mesh between processes using MPI
  
  class MPIManager
  {
  public:
    
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

    /// Finalize MPI
    static void finalize();
  private:
    /// Initialize MPI
    static void init();

    /// Constructor
    MPIManager();

    /// Destructor
    ~MPIManager();

    static MPIManager mpi;
  };
}

#endif
