// Copyright (C) 2015 Chris Richardson
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

#ifndef __MESH_RELATION_H
#define __MESH_RELATION_H

#include <vector>
#include <memory>

#include "MeshHierarchy.h"

namespace dolfin
{
  class Mesh;

  /// MeshRelation encapsulates the relationships which may exist between two Meshes
  /// or which may exist in a Mesh as a result of being related to another Mesh

  class MeshRelation
  {
  public:
    /// Constructor
    MeshRelation()
    {}

    /// Destructor
    ~MeshRelation()
    {}

  private:

    friend class MeshHierarchy;
    friend class PlazaRefinementND;

    // Map from edge of parent Mesh to new vertex in child Mesh
    // as calculated during ParallelRefinement process
    std::shared_ptr<const std::map<std::size_t, std::size_t> > edge_to_global_vertex;

  };
}

#endif
