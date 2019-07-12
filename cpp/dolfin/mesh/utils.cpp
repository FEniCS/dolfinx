// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <cstdlib>
#include <stdexcept>

using namespace dolfin;

//-----------------------------------------------------------------------------
std::string mesh::to_string(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return "point";
  case mesh::CellType::interval:
    return "interval";
  case mesh::CellType::triangle:
    return "triangle";
  case mesh::CellType::tetrahedron:
    return "tetrahedron";
  case mesh::CellType::quadrilateral:
    return "quadrilateral";
  case mesh::CellType::hexahedron:
    return "hexahedron";
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return "";
}
//-----------------------------------------------------------------------------
mesh::CellType mesh::to_type(std::string type)
{
  if (type == "point")
    return mesh::CellType::point;
  else if (type == "interval")
    return mesh::CellType::interval;
  else if (type == "triangle")
    return mesh::CellType::triangle;
  else if (type == "tetrahedron")
    return mesh::CellType::tetrahedron;
  else if (type == "quadrilateral")
    return mesh::CellType::quadrilateral;
  else if (type == "hexahedron")
    return mesh::CellType::hexahedron;
  else
    throw std::runtime_error("Unknown cell type (" + type + ")");

  // Should no reach this point
  return mesh::CellType::interval;
}
//-----------------------------------------------------------------------------
int mesh::cell_dim(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return 0;
  case mesh::CellType::interval:
    return 1;
  case mesh::CellType::triangle:
    return 2;
  case mesh::CellType::tetrahedron:
    return 3;
  case mesh::CellType::quadrilateral:
    return 2;
  case mesh::CellType::hexahedron:
    return 3;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return -1;
}
//-----------------------------------------------------------------------------
int mesh::cell_num_entities(mesh::CellType type, int dim)
{
  switch (type)
  {
  case mesh::CellType::point:
    switch (dim)
    {
    case 0:
      return 1; // vertices
    }
  case mesh::CellType::interval:
    switch (dim)
    {
    case 0:
      return 2; // vertices
    case 1:
      return 1; // cells
    }
  case mesh::CellType::triangle:
    switch (dim)
    {
    case 0:
      return 3; // vertices
    case 1:
      return 3; // edges
    case 2:
      return 1; // cells
    }
  case mesh::CellType::tetrahedron:
    switch (dim)
    {
    case 0:
      return 4; // vertices
    case 1:
      return 6; // edges
    case 2:
      return 4; // faces
    case 3:
      return 1; // cells
    }
  case mesh::CellType::quadrilateral:
    switch (dim)
    {
    case 0:
      return 4; // vertices
    case 1:
      return 4; // edges
    case 2:
      return 1; // cells
    }
  case mesh::CellType::hexahedron:
    switch (dim)
    {
    case 0:
      return 8; // vertices
    case 1:
      return 12; // edges
    case 2:
      return 6; // faces
    case 3:
      return 1; // cells
    }
  default:
    throw std::runtime_error("Unknown cell type.");
    return -1;
  }
}
//-----------------------------------------------------------------------------
bool mesh::is_simplex(mesh::CellType type)
{
  return static_cast<int>(type) > 0;
}
//-----------------------------------------------------------------------------
int mesh::num_cell_vertices(mesh::CellType type)
{
  return std::abs(static_cast<int>(type));
}
//-----------------------------------------------------------------------------
std::vector<std::int8_t> mesh::vtk_mapping(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return {0};
  case mesh::CellType::interval:
    return {0, 1};
  case mesh::CellType::triangle:
    return {0, 1, 2};
  case mesh::CellType::tetrahedron:
    return {0, 1, 2, 3};
  case mesh::CellType::quadrilateral:
    return {0, 1, 3, 2};
  case mesh::CellType::hexahedron:
    return {0, 1, 3, 2, 4, 5, 7, 6};
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return std::vector<std::int8_t>();
}
//-----------------------------------------------------------------------------
