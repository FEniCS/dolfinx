// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ReferenceCellTopology.h"
#include <cassert>
#include <stdexcept>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
const ReferenceCellTopology::Point*
ReferenceCellTopology::get_vertices(mesh::CellType cell_type)
{
  static const double interval[][3] = {{0.0}, {1.0}};
  static const double triangle[][3] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  static const double quadrilateral[][3]
      = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  static const double tetrahedron[][3]
      = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  static const double hexahedron[][3]
      = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 1.0},
         {1.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}};

  switch (cell_type)
  {
  case mesh::CellType::point:
    return nullptr;
  case mesh::CellType::interval:
    return interval;
  case mesh::CellType::triangle:
    return triangle;
  case mesh::CellType::quadrilateral:
    return quadrilateral;
  case mesh::CellType::tetrahedron:
    return tetrahedron;
  case mesh::CellType::hexahedron:
    return hexahedron;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return nullptr;
}
//-----------------------------------------------------------------------------
