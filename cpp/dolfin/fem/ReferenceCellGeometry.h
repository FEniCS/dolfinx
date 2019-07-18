// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/mesh/cell_types.h>
#include <map>
#include <set>
#include <vector>

namespace dolfin
{

namespace fem
{

class ReferenceCellGeometry
{
public:
  typedef double Point[3];

  /// Get geometric points for all vertices
  static const Point* get_vertices(mesh::CellType cell_type);
};
} // namespace fem
} // namespace dolfin
