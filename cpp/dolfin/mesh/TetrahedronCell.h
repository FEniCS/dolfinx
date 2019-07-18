// Copyright (C) 2006-2017 Anders Logg
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

/// This class implements functionality for tetrahedral cell meshes.

class TetrahedronCell : public CellTypeOld
{
public:
  /// Specify cell type and facet type
  TetrahedronCell() : mesh::CellTypeOld(CellType::tetrahedron) {}

};
} // namespace mesh
} // namespace dolfin
