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

namespace dolfin
{

  class Mesh;

  /// This class provides various funtionality for working with
  /// distributed meshes.

  class MeshDistributed
  {
  public:

    /// Find processes that own or share a vector of mesh entities (using
    /// entity global indices). Returns
    /// (global_dof, set(process_num, local_index)). Exclusively local
    /// entities will not appear in the map. Works only for vertices and
    /// cells
    static std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t> > >
    off_process_indices(const std::vector<std::size_t>& entity_indices, std::size_t dim,
                        const Mesh& mesh);

  };

}

#endif
