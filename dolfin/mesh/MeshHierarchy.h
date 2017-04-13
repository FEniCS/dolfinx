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

#ifndef __MESH_HIERARCHY_H
#define __MESH_HIERARCHY_H

#include<vector>
#include<memory>
#include<dolfin/log/log.h>

namespace dolfin
{
  class Mesh;
  class MeshRelation;
  template <typename T> class MeshFunction;

  /// Experimental implementation of a list of Meshes as a hierarchy

  class MeshHierarchy
  {
  public:
    /// Constructor
    MeshHierarchy()
    {}

    /// Constructor with initial mesh
    explicit MeshHierarchy(std::shared_ptr<const Mesh> mesh)
      : _meshes(1, mesh), _parent(NULL), _relation(NULL)
    {}

    /// Destructor
    ~MeshHierarchy()
    {}

    /// Number of meshes
    unsigned int size() const
    { return _meshes.size(); }

    /// Get Mesh i, in range [0:size()] where 0 is the coarsest Mesh.
    std::shared_ptr<const Mesh> operator[](int i) const
    {
      if (i < 0)
        i += _meshes.size();
      dolfin_assert(i < (int)_meshes.size());
        return _meshes[i];
    }

    /// Get the finest mesh of the MeshHierarchy
    std::shared_ptr<const Mesh> finest() const
    { return _meshes.back();  }

    /// Get the coarsest mesh of the MeshHierarchy
    std::shared_ptr<const Mesh> coarsest() const
    { return _meshes.front();  }

    /// Refine finest mesh of existing hierarchy, creating a new hierarchy
    /// (level n -> n+1)
    std::shared_ptr<const MeshHierarchy> refine
      (const MeshFunction<bool>& markers) const;

    /// Unrefine by returning the previous MeshHierarchy
    /// (level n -> n-1)
    /// Returns NULL for a MeshHierarchy containing a single Mesh
    std::shared_ptr<const MeshHierarchy> unrefine() const
    { return _parent; }

    /// Coarsen finest mesh by one level, based on markers (level n->n)
    std::shared_ptr<const MeshHierarchy> coarsen
      (const MeshFunction<bool>& markers) const;

    /// Calculate the number of cells on the finest Mesh
    /// which are descendents of each cell on the coarsest Mesh,
    /// returning a vector over the cells of the coarsest Mesh.
    std::vector<std::size_t> weight() const;

    /// Rebalance across processes
    std::shared_ptr<Mesh> rebalance() const;

  private:

    // Basic store of mesh pointers for easy access
    std::vector<std::shared_ptr<const Mesh> > _meshes;

    // Parent MeshHierarchy
    std::shared_ptr<const MeshHierarchy> _parent;

    // Intermesh relationship data, i.e. parent cell, facet, vertex mappings
    // instead of using MeshData
    std::shared_ptr<const MeshRelation> _relation;

  };
}

#endif
