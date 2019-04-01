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
int ReferenceCellTopology::dim(CellType cell_type)
{
  switch (cell_type)
  {
  case CellType::point:
    return 0;
  case CellType::interval:
    return 1;
  case CellType::triangle:
    return 2;
  case CellType::quadrilateral:
    return 2;
  case CellType::tetrahedron:
    return 3;
  case CellType::hexahedron:
    return 3;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return -1;
}
//-----------------------------------------------------------------------------
const int* ReferenceCellTopology::num_entities(CellType cell_type)
{
  static const int point[4] = {1, 0, 0, 0};
  static const int interval[4] = {2, 1, 0, 0};
  static const int triangle[4] = {3, 3, 1, 0};
  static const int quadrilateral[4] = {4, 4, 1, 0};
  static const int tetrahedron[4] = {4, 6, 4, 1};
  static const int hexahedron[4] = {8, 12, 6, 1};

  switch (cell_type)
  {
  case CellType::point:
    return point;
  case CellType::interval:
    return interval;
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
//---------------------------------------------------------------------
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
int ReferenceCellTopology::num_faces(CellType cell_type)
{
  switch (cell_type)
  {
  case CellType::point:
    return 0;
  case CellType::interval:
    return 0;
  case CellType::triangle:
    return 0;
  case CellType::quadrilateral:
    return 0;
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
CellType ReferenceCellTopology::entity_type(CellType cell_type, int dim, int k)
{
  switch (cell_type)
  {
  case CellType::point:
    switch (dim)
    {
    case 0:
      return CellType::point;
    }
  case CellType::interval:
    switch (dim)
    {
    case 0:
      return CellType::point;
    case 1:
      return CellType::interval;
    }
  case CellType::triangle:
    switch (dim)
    {
    case 0:
      return CellType::point;
    case 1:
      return CellType::interval;
    case 2:
      return CellType::triangle;
    }
  case CellType::quadrilateral:
    switch (dim)
    {
    case 0:
      return CellType::point;
    case 1:
      return CellType::interval;
    case 2:
      return CellType::quadrilateral;
    }
  case CellType::tetrahedron:
    switch (dim)
    {
    case 0:
      return CellType::point;
    case 1:
      return CellType::interval;
    case 2:
      return CellType::triangle;
    case 3:
      return CellType::tetrahedron;
    }
  case CellType::hexahedron:
    switch (dim)
    {
    case 0:
      return CellType::point;
    case 1:
      return CellType::interval;
    case 2:
      return CellType::quadrilateral;
    case 3:
      return CellType::hexahedron;
    }
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  throw std::runtime_error("Failed to get sub-cell type.");
  return CellType::point;
}
//-----------------------------------------------------------------------------
CellType ReferenceCellTopology::facet_type(CellType cell_type, int k)
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
ReferenceCellTopology::get_edge_vertices(CellType cell_type)
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
const ReferenceCellTopology::Face*
ReferenceCellTopology::get_face_vertices(CellType cell_type)
{
  static const int tetrahedron[][4]
      = {{1, 2, 3, -1}, {0, 2, 3, -1}, {0, 1, 3, -1}, {0, 1, 2, -1}};
  static const int hexahedron[][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5},
                                      {2, 3, 6, 7}, {0, 2, 4, 6}, {1, 3, 5, 7}};

  switch (cell_type)
  {
  case CellType::point:
    return nullptr;
  case CellType::interval:
    return nullptr;
  case CellType::triangle:
    return nullptr;
  case CellType::quadrilateral:
    return nullptr;
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
const ReferenceCellTopology::Face*
ReferenceCellTopology::get_face_edges(CellType cell_type)
{
  static const int triangle[][4] = {{0, 1, 2, -1}, };
  static const int tetrahedron[][4]
      = {{0, 1, 2, -1}, {0, 3, 4, -1}, {1, 3, 5, -1}, {2, 4, 5, -1}};
  static const int quadrilateral[][4] = {{0, 3, 1, 2}};
  static const int hexahedron[][4]
      = {{0, 1, 4, 5},   {2, 3, 6, 7},  {0, 2, 8, 9},
         {1, 3, 10, 11}, {4, 6, 8, 10}, {5, 7, 9, 11}};

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
// const int* ReferenceCellTopology::get_entities(CellType cell_type, int d0,
//                                                int d1)
// {
//   // Tetrahedron face-edge connectivity
//   static const int tetrahedron_fe[4][3]
//       = {{0, 1, 2}, {0, 3, 4}, {1, 3, 5}, {2, 4, 5}};

//   // FIXME: fill
//   static const int hexahedron_fe[6][4] = {0};

//   if (d0 == 2 and d1 == 0)
//     return (int*)get_faces(cell_type);
//   else if (d0 == 1 and d1 == 0)
//     return (int*)get_edges(cell_type);
//   else if (cell_type == CellType::tetrahedron and d0 == 2 and d1 == 1)
//     return (int*)tetrahedron_fe;
//   else if (cell_type == CellType::hexahedron and d0 == 2 and d1 == 1)
//     return (int*)hexahedron_fe;

//   return nullptr;
// }
//-----------------------------------------------------------------------------
const ReferenceCellTopology::Point*
ReferenceCellTopology::get_vertices(CellType cell_type)
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
  case CellType::point:
    return nullptr;
  case CellType::interval:
    return interval;
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
