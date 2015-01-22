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
  template <typename T> class MeshFunction;

  class MeshHierarchy
  {
  public:
    /// Constructor
    MeshHierarchy()
    {}

    /// Constructor with initial mesh
    explicit MeshHierarchy(std::shared_ptr<const Mesh> mesh)
      : _meshes(1, mesh), _parent(NULL)
    {}

    /// Destructor
    ~MeshHierarchy()
    {}

    /// Number of meshes
    unsigned int size() const
    { return _meshes.size(); }

    /// Get shared pointer to mesh i
    std::shared_ptr<const Mesh> operator[](int i) const
    {
      if (i < 0)
        i += _meshes.size();
      dolfin_assert(i < (int)_meshes.size());
        return _meshes[i];
    }

    /// Refine finest mesh of existing hierarchy, creating a new hierarchy
    //    void refine(MeshHierarchy& refined_mesh_hierarchy,
    //                const MeshFunction<bool>& markers) const;

    /// Refine finest mesh of existing hierarchy, creating a new hierarchy
    std::shared_ptr<const MeshHierarchy> refine(
                const MeshFunction<bool>& markers) const;

    /// Unrefine by returning the previous MeshHierarchy (if possible).
    /// Returns NULL for a MeshHierarchy containing a single Mesh
    std::shared_ptr<const MeshHierarchy> unrefine() const
    { return _parent; }

    /// Experiment/debug with coarsening algorithms
    void coarsen(const MeshFunction<bool>& markers);

  private:

    // Basic store of mesh pointers for easy access
    std::vector<std::shared_ptr<const Mesh> > _meshes;

    // Parent MeshHierarchy
    std::shared_ptr<const MeshHierarchy> _parent;

    // Intermesh relationship data
    // i.e. parent-child, child-parent etc. for given topological
    // dimensions, could be "parent cell-cell" or "child facet-cell"
    // etc.

    // Map from new vertices at this level to any other vertices
    // at the same level which cannot be removed until this vertex
    // is removed. Uses local indices.
    std::map<std::size_t, std::vector<std::size_t> > vertex_lock;

  };
}

#endif
