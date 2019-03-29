// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/types.h>
#include <set>
#include <memory>
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
  /// Build dofmap. The constrained domain may be a null pointer, in
  /// which case it is ignored.
  ///
  /// @param[out] dofmap
  /// @param[in] dolfin_mesh
  static std::tuple<std::size_t, std::unique_ptr<common::IndexMap>,
                    std::unordered_map<int, std::vector<int>>, std::set<int>,
                    std::vector<PetscInt>>
  build(const ElementDofLayout& el_dm, const mesh::Mesh& dolfin_mesh);

  /// Build sub-dofmap. This is a view into the parent dofmap.
  ///
  /// @param[out] sub_dofmap
  /// @param[in] parent_dofmap
  /// @param[in] component
  /// @param[in] mesh
  static std::tuple<std::int64_t, std::vector<PetscInt>> build_sub_map_view(
      const DofMap& parent_dofmap, const ElementDofLayout& parent_element_dofmap,
      const std::vector<std::size_t>& component, const mesh::Mesh& mesh);

};
} // namespace fem
} // namespace dolfin
