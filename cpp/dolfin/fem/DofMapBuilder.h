// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/fem/DofMap.h>
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

  /// Build dofmap
  static std::tuple<std::unique_ptr<common::IndexMap>, std::vector<PetscInt>>
  build(const mesh::Mesh& mesh, const ElementDofLayout& element_dof_layout,
        const std::int32_t block_size);
};
} // namespace fem
} // namespace dolfin
