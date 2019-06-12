// Copyright (C) 2006-2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

namespace dolfin
{

namespace mesh
{
class Mesh;

/// This class implements a set of basic algorithms that automate the
/// computation of mesh entities and connectivity.

class TopologyComputation
{
public:
  /// Compute mesh entities of given topological dimension by computing
  /// entity-to-vertex connectivity (dim, 0), and cell-to-entity
  /// connectivity (tdim, dim)
  static void compute_entities(Mesh& mesh, int dim);

  /// Compute connectivity (d0, d1) for given pair of topological
  /// dimensions
  static void compute_connectivity(Mesh& mesh, int d0, int d1);
};
} // namespace mesh
} // namespace dolfin
