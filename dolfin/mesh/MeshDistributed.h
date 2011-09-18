// Copyright (C) 2011 Garth N. Wells
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
// First added:  2011-09-17
// Last changed:

#ifndef __MESH_DISTRIBUTED_H
#define __MESH_DISTRIBUTED_H

#include <map>
#include <set>
#include <utility>
#include <vector>
#include <dolfin/common/types.h>
#include "MeshFunction.h"

namespace dolfin
{

  class Mesh;

  /// This class provides various funtionality for working with
  /// distributed meshes.

  class MeshDistributed
  {
  public:

    /// Find processes that own or share list of mesh entities (using
    /// entity global indices). Returns
    /// (global_dof, set(process_num, local_index)). Exclusively local
    /// entities will not appear in the map. Works only for vertices and
    /// cells
    static std::map<uint, std::set<std::pair<uint, uint> > >
    off_process_indices(const std::vector<uint>& entity_indices, uint dim,
                        const Mesh& mesh);

    /*
    /// Find processes that own or share list of mesh entities (using
    /// entity global indices). Returns
    /// (global_cell_dof, set(process_num, local_index)). Exclusively local
    /// entities will not appear in the map.
    static std::map<uint, std::set<std::pair<uint, uint> > >
    host_processes(const std::vector<std::pair<uint, uint> >& entity_indices,
                   uint dim, const Mesh& mesh);
    */

    /// Create MeshFunction from collection of pairs (global entity, value)
    template<typename T>
    static MeshFunction<T>
    create_mesh_function(const std::vector<std::pair<uint, T> >& entity_indices,
                         uint dim, const Mesh& mesh);
  };

  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>
  MeshDistributed::create_mesh_function(const std::vector<std::pair<uint, T> >& entity_indices,
                                        uint dim, const Mesh& mesh)
  {
    error("MeshDistributed::create_mesh_function not implemented");

    MeshFunction<T> mesh_function(mesh, dim);
    return mesh_function;
  }
  //---------------------------------------------------------------------------

}

#endif
