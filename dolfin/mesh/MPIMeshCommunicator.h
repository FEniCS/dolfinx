// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by: Magnus Vikstrøm, 2007.
// Modified by: Niclas Jansson, 2008.
//
// First added:  2007-05-30
// Last changed: 2008-09-16

#ifndef __MPI_MESH_COMMUNICATOR_H
#define __MPI_MESH_COMMUNICATOR_H

#include "MeshFunction.h"
#include "Mesh.h"

namespace dolfin
{

  //  class Mesh;
  
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

    /// Distribute mesh according to a mesh function                             
    static void distribute(Mesh& mesh, MeshFunction<uint>& distribution);        
                                                                                 
    /// Distribute mesh according to mesh function and preserve cell markers     
    static void distribute(Mesh& mesh, MeshFunction<uint>& distribution,         
                           MeshFunction<bool>& old_cell_marker,                  
                           MeshFunction<bool>& cell_marker);                     
                                                                                 
  private:  
                                                                     
    static void distributeCommon(Mesh& mesh, MeshFunction<uint>& distribution,   
                                 MeshFunction<bool>* old_cell_marker,            
                                 MeshFunction<bool>* cell_marker);       

  };
}

#endif
