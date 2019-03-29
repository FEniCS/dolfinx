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
const ReferenceCellTopology::Edge*
ReferenceCellTopology::get_edges(CellType cell_type)
{
  static const int triangle[][2] = {{1, 2}, {0, 2}, {0, 1}};
  static const int quadrilateral[][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};
  static const int tetrahedron[][2]
      = {{2, 3}, {1, 3}, {1, 2}, {0, 3}, {0, 2}, {0, 1}};
  static const int hexahedron[][2]
      = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3},
         {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

  switch (cell_type)
  {
  case CellType::point:
    return nullptr;
  case CellType::interval:
    return nullptr;
  case CellType::triangle:
    return triangle;
  case CellType::quadrilateral:
    return quadrilateral;
  case CellType::tetrahedron:
    return tetrahedron;
  case CellType::hexahedron:
    return hexahedron;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return nullptr;
}
//-----------------------------------------------------------------------------
