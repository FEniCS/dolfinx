// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "cell_types.h"
#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <memory>
#include <tuple>

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
class Topology;

namespace storage
{
class TopologyStorage;
}

// TODO: Make a namespace? It should not be part of the public topology
// interface probably, i.e. at least marked as internal.
// TODO: Let topology lock itself such that calls to their getter will return
// simply the data upon recursion (thread safety issues).

/// This class implements a set of basic algorithms that automate the
/// computation of mesh entities and connectivity. Members of this class should
/// should not call back on the corresponding getters of topology. In order to
/// find out whether data is present, investigate the actual storage via
/// Topology::storage.
/// All members should check whether the requiered data is already present and
/// if it this this simply return the stored data.

class TopologyComputation
{
public:
  /// Compute mesh entities of given topological dimension by computing
  /// entity-to-vertex connectivity (dim, 0), and cell-to-entity
  /// connectivity (tdim, dim)
  /// @param[in] topology Mesh topology
  /// @param[in] dim The dimension of the entities to create
  /// @return Tuple of (cell-entity connectivity, entity-vertex
  ///   connectivity, index map). If the entities already exist, then
  ///   {nullptr, nullptr, nullptr} is returned.
  static std::tuple<std::shared_ptr<const graph::AdjacencyList<std::int32_t>>,
                    std::shared_ptr<const graph::AdjacencyList<std::int32_t>>,
                    std::shared_ptr<const common::IndexMap>>
  compute_entities(const Topology& topology, int dim);


  /// Compute connectivity (d0 -> d1) for given pair of topological
  /// dimensions.
  /// Not that it is cheaper to compute first the connectivity for
  /// (max(d0, d1), min(d0, d1)) if both variants are required.
  /// @param[in] topology The topology
  /// @param[in] d0 The dimension of the nodes in the adjacency list
  /// @param[in] d1 The dimension of the edges in the adjacency list
  /// @returns The connectivities [(d0, d1), (d1, d0)] if they are
  ///   computed. If (d0, d1) already exists then a nullptr is returned.
  ///   If (d0, d1) is computed and the computation of (d1, d0) was
  ///   required as part of computing (d0, d1), the (d1, d0) is returned
  ///   as the second entry. The second entry is otherwise nullptr.
  static std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
  compute_connectivity(const Topology& topology, int d0, int d1);

  /// Compute marker for owned facets that are interior, i.e. are
  /// connected to two cells, one of which might be on a remote process.
  /// @param[in] topology The topology.
  /// @return Vector with length equal to the number of facets on this
  ///   this process. True if the ith facet (local index) is interior to
  ///   the domain.
  static std::shared_ptr<const std::vector<bool>>
  compute_interior_facets(const Topology& topology);
};

} // namespace mesh
} // namespace dolfinx
