// Copyright (C) 2006-2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>

namespace dolfin
{

class Mesh;

/// This class implements a set of basic algorithms that automate
/// the computation of mesh entities and connectivity.

class TopologyComputation
{
public:
  /// Compute mesh entities of given topological dimension, and connectivity
  /// cell-to-enity (tdim, dim)
  static std::size_t compute_entities(Mesh& mesh, std::size_t dim);

  /// Compute connectivity for given pair of topological dimensions
  static void compute_connectivity(Mesh& mesh, std::size_t d0, std::size_t d1);

private:
  // cell-to-enity (tdim, dim). Ths functions builds a list of all entities of
  // Compute mesh entities of given topological dimension, and connectivity
  // dimension dim for every cell, keyed by the sorted lists of vertex indices
  // that make up the entity. Also attached is whether or nor the entity is a
  // ghost, the local entity index (relative to the generating cell) and the
  // generating cell index. This list is then sorted, with matching keys
  // corresponding to a single enity. The entities are numbered such that ghost
  // entities come after al regular enrities.
  //
  // Returns the number of entities
  //
  // The function is templated over the number of vertices that make up an
  // entity of dimension dim. This avoid dynamic memoryt allocations, yielding
  // significant performance improvements
  template <int N>
  static std::int32_t compute_entities_by_key_matching(Mesh& mesh, int dim);

  // Compute connectivity from transpose
  static void compute_from_transpose(Mesh& mesh, std::size_t d0,
                                     std::size_t d1);

  // Direct lookup of entity from vertices in a map
  static void compute_from_map(Mesh& mesh, std::size_t d0, std::size_t d1);
};
}
