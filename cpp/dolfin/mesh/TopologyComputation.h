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

namespace mesh
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
};
} // namespace mesh
} // namespace dolfin
