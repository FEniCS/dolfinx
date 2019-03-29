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
int ReferenceCellTopology::num_edges(CellType cell_type)
{
  switch (cell_type)
  {
  case CellType::point:
    return 0;
  case CellType::interval:
    return 0;
  case CellType::triangle:
    return 3;
  case CellType::quadrilateral:
    return 4;
  case CellType::tetrahedron:
    return 6;
  case CellType::hexahedron:
    return 12;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return -1;
}
//-----------------------------------------------------------------------------
int ReferenceCellTopology::num_facets(CellType cell_type)
{
  switch (cell_type)
  {
  case CellType::point:
    return 0;
  case CellType::interval:
    return 2;
  case CellType::triangle:
    return 3;
  case CellType::quadrilateral:
    return 4;
  case CellType::tetrahedron:
    return 4;
  case CellType::hexahedron:
    return 6;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return -1;
}
//-----------------------------------------------------------------------------
CellType ReferenceCellTopology::facet_type(CellType cell_type)
{
  switch (cell_type)
  {
  case CellType::point:
    return CellType::point;
  case CellType::interval:
    return CellType::point;
  case CellType::triangle:
    return CellType::interval;
  case CellType::quadrilateral:
    return CellType::interval;
  case CellType::tetrahedron:
    return CellType::triangle;
  case CellType::hexahedron:
    return CellType::quadrilateral;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return CellType::point;
}
//-----------------------------------------------------------------------------
