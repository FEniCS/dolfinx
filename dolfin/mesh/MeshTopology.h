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
// Last changed: 2014-07-02

#ifndef __MESH_TOPOLOGY_H
#define __MESH_TOPOLOGY_H

#include <cstdint>
#include <map>
#include <utility>
#include <vector>
#include "MeshConnectivity.h"

namespace dolfin
{

  /// MeshTopology stores the topology of a mesh, consisting of mesh
  /// entities and connectivity (incidence relations for the mesh
  /// entities). Note that the mesh entities don't need to be stored,
  /// only the number of entities and the connectivity. Any numbering
  /// scheme for the mesh entities is stored separately in a
  /// MeshFunction over the entities.
  ///
  /// A mesh entity e may be identified globally as a pair e = (dim,
  /// i), where dim is the topological dimension and i is the index of
  /// the entity within that topological dimension.

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
    MeshTopology& operator= (const MeshTopology& topology);

    /// Return topological dimension
    std::size_t dim() const;

    /// Return number of entities for given dimension
    std::size_t size(std::size_t dim) const;

    /// Return global number of entities for given dimension
    std::size_t size_global(std::size_t dim) const;

    /// Return number of regular (non-ghost) entities
    /// or equivalently, the offset of where ghost entities begin
    std::size_t ghost_offset(std::size_t dim) const;

    /// Clear all data
    void clear();

    /// Clear data for given pair of topological dimensions
    void clear(std::size_t d0, std::size_t d1);

    /// Initialize topology of given maximum dimension
    void init(std::size_t dim);

    /// Set number of local entities (local_size) and global entities
    /// (global_size) for given topological dimension dim
    void init(std::size_t dim, std::size_t local_size, std::size_t global_size);

    /// Initialize storage for global entity numbering for entities of
    /// dimension dim
    void init_global_indices(std::size_t dim, std::size_t size);

    /// Initialise the offset index of ghost entities for this dimension
    void init_ghost(std::size_t dim, std::size_t index);

    /// Set global index for entity of dimension dim and with local
    /// index
    void set_global_index(std::size_t dim, std::size_t local_index,
                          std::size_t global_index)
    {
      dolfin_assert(dim < _global_indices.size());
      dolfin_assert(local_index < _global_indices[dim].size());
      _global_indices[dim][local_index] = global_index;
    }

    /// Get local-to-global index map for entities of topological
    /// dimension d
    const std::vector<std::size_t>& global_indices(std::size_t d) const
    {
      dolfin_assert(d < _global_indices.size());
      return _global_indices[d];
    }

    /// Check if global indices are available for entities of
    /// dimension dim
    bool have_global_indices(std::size_t dim) const
    {
      dolfin_assert(dim < _global_indices.size());
      return !_global_indices[dim].empty();
    }

    /// Check whether there are any shared entities calculated
    /// of dimension dim
    bool have_shared_entities(unsigned int dim) const
    { return (_shared_entities.find(dim) != _shared_entities.end()); }

    /// Return map from shared entities (local index) to processes
    /// that share the entity
    std::map<std::int32_t, std::set<unsigned int> >&
      shared_entities(unsigned int dim);

    /// Return map from shared entities (local index) to process that
    /// share the entity (const version)
    const std::map<std::int32_t, std::set<unsigned int> >&
      shared_entities(unsigned int dim) const;

    /// Return mapping from local ghost cell index to owning process
    /// Since ghost cells are at the end of the range, this is just
    /// a vector over those cells
    std::vector<unsigned int>& cell_owner()
    { return _cell_owner;  }

    /// Return mapping from local ghost cell index to owning process (const version)
    /// Since ghost cells are at the end of the range, this is just
    /// a vector over those cells
    const std::vector<unsigned int>& cell_owner() const
    { return _cell_owner;  }

    /// Return connectivity for given pair of topological dimensions
    dolfin::MeshConnectivity& operator() (std::size_t d0, std::size_t d1);

    /// Return connectivity for given pair of topological dimensions
    const dolfin::MeshConnectivity& operator() (std::size_t d0,
                                                std::size_t d1) const;

    /// Return hash based on the hash of cell-vertex connectivity
    size_t hash() const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Mesh entity colors, if computed. First vector is
    ///
    ///    (colored entity dim - dim1 - dim2 - ... -  colored entity dim)
    ///
    /// The first vector in the pair stores mesh entity colors and the
    /// vector<vector> is a list of all mesh entity indices of the same
    /// color, e.g. vector<vector>[col][i] is the index of the ith entity
    /// of color 'col'.
    // Developer note: std::vector is used in place of a MeshFunction
    //                 to avoid circular dependencies in the header files
    std::map<std::vector<std::size_t>,
      std::pair<std::vector<std::size_t>,
      std::vector<std::vector<std::size_t>>>> coloring;

  private:

    // Number of mesh entities for each topological dimension
    std::vector<unsigned int> num_entities;

    // Number of ghost indices for each topological dimension
    std::vector<std::size_t> ghost_offset_index;

    // Global number of mesh entities for each topological dimension
    std::vector<std::size_t> global_num_entities;

    // Global indices for mesh entities (empty if not set)
    std::vector<std::vector<std::size_t> > _global_indices;

    // For entities of a given dimension d , maps each shared entity
    // (local index) to a list of the processes sharing the vertex
    std::map<unsigned int, std::map<std::int32_t, std::set<unsigned int>>>
      _shared_entities;

    // For cells which are "ghosted", locate the owning process,
    // using a vector rather than a map,
    // since ghost cells are always at the end of the range.
    std::vector<unsigned int> _cell_owner;

    // Connectivity for pairs of topological dimensions
    std::vector<std::vector<MeshConnectivity> > connectivity;

  };

}

#endif
