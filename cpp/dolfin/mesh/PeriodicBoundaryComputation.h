// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/constants.h>
#include <dolfin/mesh/MeshFunction.h>
#include <map>
#include <utility>
#include <vector>

namespace dolfin
{

class Mesh;

namespace mesh
{
class SubDomain;
}

/// This class computes map from slave entity to master entity

class PeriodicBoundaryComputation
{
public:
  /// For entities of dimension dim, compute map from a slave entity
  /// on this process (local index) to its master entity (owning
  /// process, local index on owner). If a master entity is shared
  /// by processes, only one of the owning processes is returned.
  static std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>>
  compute_periodic_pairs(const Mesh& mesh, const mesh::SubDomain& sub_domain,
                         const std::size_t dim);

  /// This function returns a MeshFunction which marks mesh entities
  /// of dimension dim according to:
  ///
  ///     2: slave entities
  ///     1: master entities
  ///     0: all other entities
  ///
  /// It is useful for visualising and debugging the Expression::map
  /// function that is used to apply periodic boundary conditions.
  static MeshFunction<std::size_t>
  masters_slaves(std::shared_ptr<const Mesh> mesh, const mesh::SubDomain& sub_domain,
                 const std::size_t dim);

private:
  // Return true is point lies within bounding box
  static bool in_bounding_box(const std::vector<double>& point,
                              const std::vector<double>& bounding_box,
                              const double tol);
};
}
