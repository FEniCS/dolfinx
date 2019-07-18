// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include <vector>

namespace dolfin
{

namespace mesh
{
class Cell;

/// This class implements functionality for hexahedral cell  meshes.

class HexahedronCell : public mesh::CellTypeOld
{
public:
  /// Specify cell type and facet type
  HexahedronCell() : mesh::CellTypeOld(CellType::hexahedron) {}

  /// Return orientation of the cell
  std::size_t orientation(const mesh::Cell& cell) const;
};
} // namespace mesh
} // namespace dolfin
