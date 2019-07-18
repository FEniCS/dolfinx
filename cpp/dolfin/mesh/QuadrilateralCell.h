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

/// This class implements functionality for quadrilateral cells.

class QuadrilateralCell : public CellTypeOld
{
public:
  /// Specify cell type and facet type
  QuadrilateralCell() : mesh::CellTypeOld(CellType::quadrilateral) {}
};
} // namespace mesh
} // namespace dolfin
