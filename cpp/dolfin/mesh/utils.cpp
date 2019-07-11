// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"

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
    throw std::runtime_error("Unknown cell type");
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
