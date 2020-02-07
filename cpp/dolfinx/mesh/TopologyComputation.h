// Copyright (C) 2006-2010 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <memory>
#include <tuple>

namespace dolfinx
{

namespace graph
{
template <typename T>
class AdjacencyList;

}

namespace mesh
{
class Mesh;

/// This class implements a set of basic algorithms that automate the
/// computation of mesh entities and connectivity

class TopologyComputation
{
public:
  /// Compute mesh entities of given topological dimension by computing
  /// entity-to-vertex connectivity (dim, 0), and cell-to-entity
  /// connectivity (tdim, dim)
  /// @param [in] mesh The mesh
  /// @param [in] dim The dimension of the entities to create
  /// @return Tuple of (cell-entity connectivity, entity-vertex
  ///   connectivity, number of created entities). The entities already
  ///   exists, then {nullptr, nullptr, -1} is returned.
  static std::tuple<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
                    std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
                    std::int32_t>
  compute_entities(const Mesh& mesh, int dim);

  /// Compute connectivity (d0, d1) for given pair of topological
  /// dimensions
  static void compute_connectivity(Mesh& mesh, int d0, int d1);
};
} // namespace mesh
} // namespace dolfinx