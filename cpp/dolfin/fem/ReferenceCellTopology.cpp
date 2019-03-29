// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ReferenceCellTopology.h"
#include <stdexcept>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
int ReferenceCellTopology::num_vertices(CellType cell_type)
{
  switch (cell_type)
  {
  case CellType::point:
    return 1;
  case CellType::interval:
    return 2;
  case CellType::triangle:
    return 3;
  case CellType::quadrilateral:
    return 4;
  case CellType::tetrahedron:
    return 4;
  case CellType::hexahedron:
    return 8;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return -1;
}
//-----------------------------------------------------------------------------
