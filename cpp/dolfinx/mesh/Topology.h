// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include <dolfinx/graph/AdjacencyList.h>

namespace dolfinx
{
namespace common
{
class IndexMap;
}

namespace graph
{
template <typename T>
class AdjacencyList;
}

namespace mesh
{

/// Topology stores the topology of a mesh, consisting of mesh entities
/// and connectivity (incidence relations for the mesh entities). Note
/// that the mesh entities don't need to be stored, only the number of
/// entities and the connectivity. Any numbering scheme for the mesh
/// entities is stored separately in a MeshFunction over the entities.
///
/// A mesh entity e may be identified globally as a pair e = (dim, i),
/// where dim is the topological dimension and i is the index of the
/// entity within that topological dimension.

class Topology
{
public:
  /// Create empty mesh topology
  Topology(int dim);

  /// Copy constructor
  Topology(const Topology& topology) = default;

  /// Move constructor
  Topology(Topology&& topology) = default;

  /// Destructor
  ~Topology() = default;

  /// Assignment
  Topology& operator=(const Topology& topology) = default;

  /// Return topological dimension
  int dim() const;

  /// @todo Remove this function. Use IndexMap instead
  /// Set the global indices for entities of dimension dim
  void set_global_indices(int dim,
                          const std::vector<std::int64_t>& global_indices);

  /// Set the IndexMap for dimension dim
  /// @warning This is experimental and likely to change
  void set_index_map(int dim,
                     std::shared_ptr<const common::IndexMap> index_map);

  /// Get the IndexMap for dimension dim
  /// (Currently partially working)
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// @todo Remove this function. Use IndexMap instead
  /// Get local-to-global index map for entities of topological
  /// dimension d
  const std::vector<std::int64_t>& global_indices(int d) const;

  /// Set the map from shared entities (local index) to processes that
  /// share the entity
  void set_shared_entities(
      int dim, const std::map<std::int32_t, std::set<std::int32_t>>& entities);

  /// @todo Remove this function
  /// Return map from shared entities (local index) to process that
  /// share the entity (const version)
  const std::map<std::int32_t, std::set<std::int32_t>>&
  shared_entities(int dim) const;

  /// Marker for entities of dimension dim on the boundary. An entity of
  /// co-dimension < 0 is on the boundary if it is connected to a boundary
  /// facet. It is not defined for codimension 0.
  /// @param[in] dim Toplogical dimension of the entities to check. It
  /// must be less than the topological dimension.
  /// @return Vector of length equal to number of local entities, with
  ///          'true' for entities on the boundary and otherwise 'false'.
  std::vector<bool> on_boundary(int dim) const;

  /// Return connectivity for given pair of topological dimensions
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> connectivity(int d0,
                                                                   int d1);

  /// Return connectivity for given pair of topological dimensions
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  connectivity(int d0, int d1) const;

  /// Set connectivity for given pair of topological dimensions
  void set_connectivity(std::shared_ptr<graph::AdjacencyList<std::int32_t>> c,
                        int d0, int d1);

  /// Return hash based on the hash of cell-vertex connectivity
  size_t hash() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

  /// @todo Move this outside of this class
  /// Set global number of connections for each local entities
  void set_global_size(std::array<int, 2> d,
                       const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>&
                           num_global_connections)
  {
    // assert(num_global_connections.size() == _offsets.size() - 1);
    _num_global_connections(d[0], d[1]) = num_global_connections;
  }

  /// @todo Can this be removed?
  /// Return global number of connections for given entity
  int size_global(std::array<int, 2> d, std::int32_t entity) const
  {
    if (_num_global_connections(d[0], d[1]).size() == 0)
      return _connectivity(d[0], d[1])->num_links(entity);
    else
      return _num_global_connections(d[0], d[1])[entity];
  }

private:
  // Global indices for mesh entities
  std::vector<std::vector<std::int64_t>> _global_indices;

  // TODO: Could IndexMap be used here in place of std::map?
  // For entities of a given dimension d, maps each shared entity
  // (local index) to a list of the processes sharing the vertex
  std::vector<std::map<std::int32_t, std::set<std::int32_t>>> _shared_entities;

  // IndexMap to store ghosting for each entity dimension
  std::array<std::shared_ptr<const common::IndexMap>, 4> _index_map;

  // AdjacencyList for pairs of topological dimensions
  Eigen::Array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
               Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _connectivity;

  // TODO: revise
  // Global number of connections for each entity (possibly not
  // computed)
  Eigen::Array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>, 4, 4>
      _num_global_connections;
};
} // namespace mesh
} // namespace dolfinx
