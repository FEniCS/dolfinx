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
    static void coarsenSimplexByNodeDeletion(Mesh& mesh, Edge& edge);

  private:

    /*
    /// Collapse edge by node deletion 
    void collapseEdgeByNodeDeletion(Edge& edge, 
                                    uint& vertex_to_remove, 
                                    MeshEditor& editor, 
                                    uint& current_cell); 
    */

  };

}

#endif
