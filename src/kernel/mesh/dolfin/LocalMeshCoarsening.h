// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-01

#ifndef __LOCAL_MESH_COARSENING_H
#define __LOCAL_MESH_COARSENING_H

#include <dolfin/constants.h>

namespace dolfin
{

  class Mesh;
  class Edge;

  /// This class implements local mesh coarsening for different mesh types.

  class LocalMeshCoarsening
  {
  public:

    /// Coarsen simplicial mesh locally by node deletion 
    static void coarsenMeshByEdgeCollapse(Mesh& mesh, 
                                          MeshFunction<bool>& cell_marker,
                                          bool coarsen_boundary = false); 

  private:

    /// Collapse edge by node deletion 
    static void collapseEdge(Mesh& mesh, Edge& edge, 
                             Vertex& vertex_to_remove, 
                             MeshFunction<bool>& cell_to_remove, 
                             Array<int>& old2new_vertex, 
                             MeshEditor& editor, 
                             uint& current_cell); 

  };

}

#endif
