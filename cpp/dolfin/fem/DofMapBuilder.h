// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/DofCellPermutations.h>
#include <memory>
#include <petscsys.h>
#include <tuple>
#include <vector>

namespace dolfin
{

namespace common
{
class IndexMap;
}

namespace mesh
{
class Mesh;
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
  build(const mesh::Mesh& mesh,
        std::shared_ptr<const ElementDofLayout> element_dof_layout);

  /// Build sub-dofmap view
  static DofMap build_submap(const DofMap& dofmap_parent,
                             const std::vector<int>& component,
                             const mesh::Mesh& mesh);

  /// Build dofmap
  static std::tuple<std::unique_ptr<common::IndexMap>,
                    Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
  build(const mesh::Mesh& mesh, const ElementDofLayout& element_dof_layout,
        const std::int32_t block_size);
};
} // namespace fem
} // namespace dolfin
