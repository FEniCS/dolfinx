// Copyright (C) 2006-2009 Anders Logg
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
// First added:  2006-05-08
// Last changed: 2011-09-01

#ifndef __MESH_TOPOLOGY_H
#define __MESH_TOPOLOGY_H

#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  class MeshConnectivity;

  /// MeshTopology stores the topology of a mesh, consisting of mesh entities
  /// and connectivity (incidence relations for the mesh entities). Note that
  /// the mesh entities don't need to be stored, only the number of entities
  /// and the connectivity. Any numbering scheme for the mesh entities is
  /// stored separately in a MeshFunction over the entities.
  ///
  /// A mesh entity e may be identified globally as a pair e = (dim, i), where
  /// dim is the topological dimension and i is the index of the entity within
  /// that topological dimension.

  class MeshTopology
  {
  public:

    /// Create empty mesh topology
    MeshTopology();

    /// Copy constructor
    MeshTopology(const MeshTopology& topology);

    /// Destructor
    ~MeshTopology();

    /// Assignment
    const MeshTopology& operator= (const MeshTopology& topology);

    /// Return topological dimension
    uint dim() const;

    /// Return number of entities for given dimension
    uint size(uint dim) const;

    /// Clear all data
    void clear();

    /// Clear data for given pair of topological dimensions
    void clear(uint d0, uint d1);

    /// Initialize topology of given maximum dimension
    void init(uint dim);

    /// Set number of entities (size) for given topological dimension
    void init(uint dim, uint size);

    /// Initialize global enity numbering for entities of dimension dim
    void init_global_indices(uint dim, uint size);

    /// Set global index for entity of dimension dim and with local index
    void set_global_index(uint dim, uint local_index, uint global_index)
    {
      dolfin_assert(dim < _global_indices.size());
      dolfin_assert(local_index < _global_indices[dim].size());
      _global_indices[dim][local_index] = global_index;
    }

    /// Get local-to-global index map for entities of topological dimension d
    const std::vector<uint>& global_indices(uint d) const
    {
      dolfin_assert(d < _global_indices.size());
      return _global_indices[d];
    }

    /// Check if global indices are available for entiries of dimension dim
    bool have_global_indices(uint dim) const
    {
      dolfin_assert(dim < _global_indices.size());
      return !_global_indices[dim].empty();
    }

    /// Return connectivity for given pair of topological dimensions
    dolfin::MeshConnectivity& operator() (uint d0, uint d1);

    /// Return connectivity for given pair of topological dimensions
    const dolfin::MeshConnectivity& operator() (uint d0, uint d1) const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    // Friends
    friend class BinaryFile;

    // Number of mesh entities for each topological dimension
    std::vector<uint> num_entities;

    // Global indices for mesh entities (empty if not set)
    std::vector<std::vector<uint> > _global_indices;

    // Connectivity for pairs of topological dimensions
    std::vector<std::vector<MeshConnectivity> > connectivity;

  };

}

#endif
