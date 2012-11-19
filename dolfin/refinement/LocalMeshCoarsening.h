// Copyright (C) 2006 Johan Hoffman
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
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
  template <typename T> class MeshFunction;

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
    static bool coarsen_mesh_ok(Mesh& mesh, std::size_t edge_index, std::size_t* edge_vertex,
                                MeshFunction<bool>& vertex_forbidden);

    /// Collapse edge by node deletion
    static void collapse_edge(Mesh& mesh, Edge& edge,
                              Vertex& vertex_to_remove,
                              MeshFunction<bool>& cell_to_remove,
                              std::vector<int>& old2new_vertex,
                              std::vector<int>& old2new_cell,
                              MeshEditor& editor,
                              std::size_t& current_cell);

    /// Coarsen simplicial cell by edge collapse
    static bool coarsen_cell(Mesh& mesh, Mesh& coarse_mesh,
                             int cell_id,
                             std::vector<int>& old2new_vertex,
                             std::vector<int>& old2new_cell,
                             bool coarsen_boundary = false);

  };

}

#endif
