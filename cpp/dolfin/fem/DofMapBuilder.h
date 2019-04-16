// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/types.h>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
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
  /// Build dofmap.
  ///
  /// @param[out] dofmap
  /// @param[in] dolfin_mesh
  static std::tuple<std::int64_t, std::unique_ptr<common::IndexMap>,
                    std::unordered_map<std::int32_t, std::vector<std::int32_t>>,
                    std::set<std::int32_t>, std::vector<PetscInt>>
  build(const mesh::Mesh& dolfin_mesh,
        const ElementDofLayout& element_dof_layout,
        const std::int32_t block_size);
};
} // namespace fem
} // namespace dolfin
