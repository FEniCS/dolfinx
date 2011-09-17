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

#include <utility>
#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  class Mesh;

  /// This class provides various funtionality for working with
  /// distributed meshes.

  class MeshDistributed
  {
  public:

    /// Find processes that own or share list of mesh entities (using
    /// entity global indices)
    static std::vector<uint>
    host_processes(const std::vector<uint> entity_indices, uint dim,
                   const Mesh& mesh);

    /// Find processes that own or share list of mesh entities (using
    /// global cell index + cell-wise entity index)
    static std::vector<uint>
    host_processes(const std::vector<std::pair<uint, uint> > entity_indices,
                   uint dim, const Mesh& mesh);

  };

}

#endif
