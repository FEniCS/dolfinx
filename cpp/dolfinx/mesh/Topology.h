// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "cell_types.h"
#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <vector>

namespace dolfinx
{
namespace common
{
class IndexMap;
}

namespace mesh
{

class Topology;

/// Compute marker for owned facets that are interior, i.e. are
/// connected to two cells, one of which might be on a remote process
/// @param[in] topology The topology
/// @return Vector with length equal to the number of facets on this
///   this process. True if the ith facet (local index) is interior to
///   the domain.
std::vector<bool> compute_interior_facets(const Topology& topology);

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
  Topology(mesh::CellType type);

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
  void
  set_global_user_vertices(const std::vector<std::int64_t>& vertex_indices);

  /// Set the IndexMap for dimension dim
  /// @warning This is experimental and likely to change
  void set_index_map(int dim,
                     std::shared_ptr<const common::IndexMap> index_map);

  /// Get the IndexMap for dimension dim
  /// (Currently partially working)
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// @todo Remove/revise this function. Use IndexMap instead.
  /// Get local-to-global index map for entities of topological
  /// dimension d
  const std::vector<std::int64_t>& get_global_user_vertices() const;

  /// Marker for entities of dimension dim on the boundary. An entity of
  /// co-dimension < 0 is on the boundary if it is connected to a
  /// boundary facet. It is not defined for codimension 0.
  /// @param[in] dim Toplogical dimension of the entities to check. It
  ///   must be less than the topological dimension.
  /// @return Vector of length equal to number of local entities, with
  ///   'true' for entities on the boundary and otherwise 'false'.
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

  /// Gets markers for owned facets that are interior, i.e. are
  /// connected to two cells, one of which might be on a remote process
  /// @return Vector with length equal to the number of facets owned by
  ///   this process. True if the ith facet (local index) is interior to
  ///   the domain.
  const std::vector<bool>& interior_facets() const;

  /// Set markers for owned facets that are interior
  /// @param[in] interior_facets The marker vector
  void set_interior_facets(const std::vector<bool>& interior_facets);

  /// Return hash based on the hash of cell-vertex connectivity
  size_t hash() const;

  /// Cell type
  /// @return Cell type that th topology is for
  mesh::CellType cell_type() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

private:
  // Cell type
  mesh::CellType _cell_type;

  // Global indices for vertices
  std::vector<std::int64_t> _global_user_vertices;

  // IndexMap to store ghosting for each entity dimension
  std::array<std::shared_ptr<const common::IndexMap>, 4> _index_map;

  // AdjacencyList for pairs of topological dimensions
  Eigen::Array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
               Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _connectivity;

  // Marker for owned facets, which evaluates to True for facets that
  // are interior to the domain
  std::shared_ptr<const std::vector<bool>> _interior_facets;
};
} // namespace mesh
} // namespace dolfinx
