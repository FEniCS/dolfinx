// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/common/types.h>
#include <memory>
#include <vector>

namespace dolfin
{

enum class CellType : short int
  {
    point,
    interval,
    triangle,
    quadrilateral,
    tetrahedron,
    hexahedron
  };


// namespace mesh
// {
// class CellType;
// }

namespace fem
{

class ReferenceCellTopology
{
public:
  static int num_vertices(CellType cell_type);

};
} // namespace fem
} // namespace dolfin
