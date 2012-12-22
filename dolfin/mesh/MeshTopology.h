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
// Last changed: 2012-10-25

#ifndef __MESH_TOPOLOGY_H
#define __MESH_TOPOLOGY_H

#include <map>
#include <utility>
#include <vector>
#include "MeshConnectivity.h"

namespace dolfin
{

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
    std::size_t dim() const;

    /// Return number of entities for given dimension
    std::size_t size(std::size_t dim) const;

    /// Return global number of entities for given dimension
    std::size_t size_global(std::size_t dim) const;

    /// Clear all data
    void clear();

    /// Clear data for given pair of topological dimensions
    void clear(std::size_t d0, std::size_t d1);

    /// Initialize topology of given maximum dimension
    void init(std::size_t dim);

    /// Set number of local entities (local_size) for given topological
    /// dimension
    void init(std::size_t dim, std::size_t local_size);

    /// Set number of global entities (global_size) for given topological
    /// dimension
    void init_global(std::size_t dim, std::size_t global_size);

    /// Initialize storage for global entity numbering for entities of
    /// dimension dim
    void init_global_indices(std::size_t dim, std::size_t size);

    /// Set global index for entity of dimension dim and with local index
    void set_global_index(std::size_t dim, std::size_t local_index, std::size_t global_index)
    {
      dolfin_assert(dim < _global_indices.size());
      dolfin_assert(local_index < _global_indices[dim].size());
      _global_indices[dim][local_index] = global_index;
    }

    /// Get local-to-global index map for entities of topological dimension d
    const std::vector<std::size_t>& global_indices(std::size_t d) const
    {
      dolfin_assert(d < _global_indices.size());
      return _global_indices[d];
    }

    /// Check if global indices are available for entiries of dimension dim
    bool have_global_indices(std::size_t dim) const
    {
      dolfin_assert(dim < _global_indices.size());
      return !_global_indices[dim].empty();
    }

    /// Return map from shared entiies to process that share the entity
    std::map<std::size_t, std::set<std::size_t> >&
      shared_entities(std::size_t dim);

    /// Return map from shared entiies to process that share the entity
    /// (const version)
    const std::map<std::size_t, std::set<std::size_t> >&
      shared_entities(std::size_t dim) const;

    /// Return connectivity for given pair of topological dimensions
    dolfin::MeshConnectivity& operator() (std::size_t d0, std::size_t d1);

    /// Return connectivity for given pair of topological dimensions
    const dolfin::MeshConnectivity& operator() (std::size_t d0, std::size_t d1) const;

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
    std::map<const std::vector<std::size_t>,
      std::pair<std::vector<std::size_t>, std::vector<std::vector<std::size_t> > > > coloring;

  private:

    // Friends
    friend class BinaryFile;

    // Number of mesh entities for each topological dimension
    std::vector<std::size_t> num_entities;

    // Global number of mesh entities for each topological dimension
    std::vector<std::size_t> global_num_entities;

    // Global indices for mesh entities (empty if not set)
    std::vector<std::vector<std::size_t> > _global_indices;

    // Maps each shared vertex (entity of dim 0) to a list of the
    // processes sharing the vertex
    std::map<std::size_t, std::set<std::size_t> > _shared_vertices;

    // Connectivity for pairs of topological dimensions
    std::vector<std::vector<MeshConnectivity> > connectivity;

  };

}

#endif
