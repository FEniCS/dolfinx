// Copyright (C) 2019 Matthew Scroggs
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/common/types.h>
#include <ufc.h>

// FIXME: This functionality should be moved into the ElementDofLayout

namespace dolfin
{
namespace mesh
{
enum class CellType;
class Mesh;
} // namespace mesh

namespace fem
{
/// Class that contains information about the dof arrangement on entities
class EntityArrangementTypes
{
public:
  /// Construction
  EntityArrangementTypes(const ufc_dofmap& dofmap,
                         const mesh::CellType& cell_type);

  /// Get the blocksize on an entity of a dimension
  /// @param[in] dim The dimension of the entity
  /// @return The block size
  int get_block_size(const int dim) const;

private:
  std::array<int, 4> _entity_block_size;
};

} // namespace fem
} // namespace dolfin
