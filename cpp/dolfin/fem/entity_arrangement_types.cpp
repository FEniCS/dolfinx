// Copyright (C) 2019 Matthew Scroggs
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "entity_arrangement_types.h"
#include <dolfin/mesh/cell_types.h>

#include <iostream>

// FIXME: This functionality should be moved into the ElementDofLayout

namespace
{
//-----------------------------------------------------------------------------
} // namespace

using namespace dolfin;

//-----------------------------------------------------------------------------
fem::EntityArrangementTypes::EntityArrangementTypes(
    const ufc_dofmap& dofmap, const mesh::CellType& cell_type)
{
  for (int i = 0; i < 4; ++i)
  {
    _entity_block_size[i] = dofmap.entity_block_size[i];
    std::cout << dofmap.entity_block_size[i] << ",";
  }
  std::cout << std::endl;
}
//-----------------------------------------------------------------------------
int fem::EntityArrangementTypes::get_block_size(const int dim) const
{
  return _entity_block_size[dim];
}
//-----------------------------------------------------------------------------
