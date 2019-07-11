// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::string mesh::to_string(mesh::CellTypeNew type)
{
  switch (type)
  {
  case mesh::CellTypeNew::point:
    return "point";
  case mesh::CellTypeNew::interval:
    return "interval";
  case mesh::CellTypeNew::triangle:
    return "triangle";
  case mesh::CellTypeNew::tetrahedron:
    return "tetrahedron";
  case mesh::CellTypeNew::quadrilateral:
    return "quadrilateral";
  case mesh::CellTypeNew::hexahedron:
    return "hexahedron";
  default:
    throw std::runtime_error("Unknown cell type");
  }

  return "";
}
//-----------------------------------------------------------------------------
mesh::CellTypeNew mesh::to_type(std::string type)
{
  if (type == "point")
    return mesh::CellTypeNew::point;
  else if (type == "interval")
    return mesh::CellTypeNew::interval;
  else if (type == "triangle")
    return mesh::CellTypeNew::triangle;
  else if (type == "tetrahedron")
    return mesh::CellTypeNew::tetrahedron;
  else if (type == "quadrilateral")
    return mesh::CellTypeNew::quadrilateral;
  else if (type == "hexahedron")
    return mesh::CellTypeNew::hexahedron;
  else
    throw std::runtime_error("Unknown cell type (" + type + ")");

  // Should no reach this point
  return mesh::CellTypeNew::interval;
}
//-----------------------------------------------------------------------------
