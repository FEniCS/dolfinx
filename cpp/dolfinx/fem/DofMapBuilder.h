// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "petscsys.h"
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/graph/AdjacencyList.h>
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
class ElementDofLayout;

/// Builds a DofMap on a mesh::Mesh

class DofMapBuilder
{

public:
  /// Build dofmap
  static std::tuple<std::shared_ptr<const ElementDofLayout>,
                    std::shared_ptr<const common::IndexMap>,
                    graph::AdjacencyList<std::int32_t>>
  build(MPI_Comm comm, const mesh::Topology& topology,
        std::shared_ptr<const ElementDofLayout> element_dof_layout);

  /// Build sub-dofmap view
  static std::tuple<std::shared_ptr<const ElementDofLayout>,
                    graph::AdjacencyList<std::int32_t>>
  build_submap(const ElementDofLayout& dof_layout_parent,
               const graph::AdjacencyList<PetscInt>& dofmap_parent,
               const std::vector<int>& component);

  /// Build dofmap
  static std::pair<std::shared_ptr<common::IndexMap>,
                   graph::AdjacencyList<std::int32_t>>
  build(MPI_Comm comm, const mesh::Topology& topology,
        const ElementDofLayout& element_dof_layout,
        const std::int32_t block_size);
};
} // namespace fem
} // namespace dolfinx
