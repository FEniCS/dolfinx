// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-11-01

#ifndef __LOCAL_MESH_COARSENING_H
#define __LOCAL_MESH_COARSENING_H

#include <vector>

namespace dolfin
{

  class Mesh;
  class Edge;
  class Vertex;
  class MeshEditor;
  template <class T> class MeshFunction;

  /// This class implements local mesh coarsening for different mesh types.

  class LocalMeshCoarsening
  {
  public:

    /// Coarsen simplicial mesh locally by edge collapse
    static void coarsen_mesh_by_edge_collapse(Mesh& mesh,
                                              MeshFunction<bool>& cell_marker,
                                              bool coarsen_boundary = false);

  private:

    /// Check that edge collapse is ok
    static bool coarsen_mesh_ok(Mesh& mesh, uint edge_index, uint* edge_vertex,
                                MeshFunction<bool>& vertex_forbidden);

    /// Collapse edge by node deletion
    static void collapse_edge(Mesh& mesh, Edge& edge,
                              Vertex& vertex_to_remove,
                              MeshFunction<bool>& cell_to_remove,
                              std::vector<int>& old2new_vertex,
                              std::vector<int>& old2new_cell,
                              MeshEditor& editor,
                              uint& current_cell);

    /// Coarsen simplicial cell by edge collapse
    static bool coarsen_cell(Mesh& mesh, Mesh& coarse_mesh,
                             int cell_id,
                             std::vector<int>& old2new_vertex,
                             std::vector<int>& old2new_cell,
                             bool coarsen_boundary = false);

  };

}

#endif
