// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/fem/DofMap.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>
#include <tuple>
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
} // namespace mesh

namespace fem
{
class DofMap;
class ElementDofLayout;

/// Builds a DofMap on a mesh::Mesh

class DofMapBuilder
{

public:
  /// Build dofmap
  static DofMap
  build(MPI_Comm comm, const mesh::Topology& topology,
        std::shared_ptr<const ElementDofLayout> element_dof_layout);

  /// Build sub-dofmap view
  static DofMap build_submap(const DofMap& dofmap_parent,
                             const std::vector<int>& component,
                             const mesh::Topology& topology);

  /// Build dofmap
  static std::pair<std::unique_ptr<common::IndexMap>,
                   Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
  build(MPI_Comm comm, const mesh::Topology& topology,
        const ElementDofLayout& element_dof_layout,
        const std::int32_t block_size);
};
} // namespace fem
} // namespace dolfinx
