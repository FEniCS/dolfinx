// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "cell_types.h"
#include "Cell.h"
#include "Facet.h"
#include "Geometry.h"
#include "MeshEntity.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <stdexcept>

using namespace dolfin;

namespace
{
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_entities_interval(int dim)
{
  assert(dim == 0);
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> e(2, 1);
  e(0, 0) = 0;
  e(1, 0) = 1;
  return e;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_entities_triangle(int dim)
{
  // We only need to know how to create edges
  assert(dim == 1);

  // Create the three edges
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> e(3, 2);
  e(0, 0) = 1;
  e(0, 1) = 2;
  e(1, 0) = 0;
  e(1, 1) = 2;
  e(2, 0) = 0;
  e(2, 1) = 1;

  return e;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_entities_quadrilateral(int dim)
{
  assert(dim == 1);

  // Create the four edges
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> e(4, 2);
  e(0, 0) = 0;
  e(0, 1) = 1;
  e(1, 0) = 2;
  e(1, 1) = 3;
  e(2, 0) = 0;
  e(2, 1) = 2;
  e(3, 0) = 1;
  e(3, 1) = 3;

  return e;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_entities_tetrahedron(int dim)
{
  // We only need to know how to create edges and faces
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> e;
  switch (dim)
  {
  case 1:
    // Resize data structure
    e.resize(6, 2);

    // Create the six edges
    e(0, 0) = 2;
    e(0, 1) = 3;
    e(1, 0) = 1;
    e(1, 1) = 3;
    e(2, 0) = 1;
    e(2, 1) = 2;
    e(3, 0) = 0;
    e(3, 1) = 3;
    e(4, 0) = 0;
    e(4, 1) = 2;
    e(5, 0) = 0;
    e(5, 1) = 1;
    break;
  case 2:
    // Resize data structure
    e.resize(4, 3);

    // Create the four faces
    e(0, 0) = 1;
    e(0, 1) = 2;
    e(0, 2) = 3;
    e(1, 0) = 0;
    e(1, 1) = 2;
    e(1, 2) = 3;
    e(2, 0) = 0;
    e(2, 1) = 1;
    e(2, 2) = 3;
    e(3, 0) = 0;
    e(3, 1) = 1;
    e(3, 2) = 2;
    break;
  default:
    throw std::runtime_error("Illegal topological dimension");
  }

  return e;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_entities_hexahedron(int dim)
{
  // We need to know how to create edges and faces

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> e;
  switch (dim)
  {
  case 1:
    // Resize data structure
    e.resize(12, 2);

    // Create the 12 edges
    e(0, 0) = 0;
    e(0, 1) = 1;
    e(1, 0) = 2;
    e(1, 1) = 3;
    e(2, 0) = 4;
    e(2, 1) = 5;
    e(3, 0) = 6;
    e(3, 1) = 7;
    e(4, 0) = 0;
    e(4, 1) = 2;
    e(5, 0) = 1;
    e(5, 1) = 3;
    e(6, 0) = 4;
    e(6, 1) = 6;
    e(7, 0) = 5;
    e(7, 1) = 7;
    e(8, 0) = 0;
    e(8, 1) = 4;
    e(9, 0) = 1;
    e(9, 1) = 5;
    e(10, 0) = 2;
    e(10, 1) = 6;
    e(11, 0) = 3;
    e(11, 1) = 7;
    break;
  case 2:
    // Resize data structure
    e.resize(6, 4);

    // Create the 6 faces
    e(0, 0) = 0;
    e(0, 1) = 1;
    e(0, 2) = 2;
    e(0, 3) = 3;
    e(1, 0) = 4;
    e(1, 1) = 5;
    e(1, 2) = 6;
    e(1, 3) = 7;
    e(2, 0) = 0;
    e(2, 1) = 1;
    e(2, 2) = 4;
    e(2, 3) = 5;
    e(3, 0) = 2;
    e(3, 1) = 3;
    e(3, 2) = 6;
    e(3, 3) = 7;
    e(4, 0) = 0;
    e(4, 1) = 2;
    e(4, 2) = 4;
    e(4, 3) = 6;
    e(5, 0) = 1;
    e(5, 1) = 3;
    e(5, 2) = 5;
    e(5, 3) = 7;
    break;
  default:
    throw std::runtime_error("Illegal topological dimension. Must be 1 or 2.");
  }

  return e;
}
//-----------------------------------------------------------------------------

} // namespace

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
    return std::string();
  }
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
mesh::CellType mesh::cell_entity_type(mesh::CellType type, int d)
{
  const int dim = mesh::cell_dim(type);
  if (d == dim)
    return type;
  else if (d == 1)
    return CellType::interval;
  else if (d == (dim - 1))
    return mesh::cell_facet_type(type);

  return CellType::point;
}
//-----------------------------------------------------------------------------
mesh::CellType mesh::cell_facet_type(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return mesh::CellType::point;
  case mesh::CellType::interval:
    return mesh::CellType::point;
  case mesh::CellType::triangle:
    return mesh::CellType::interval;
  case mesh::CellType::tetrahedron:
    return mesh::CellType::triangle;
  case mesh::CellType::quadrilateral:
    return mesh::CellType::interval;
  case mesh::CellType::hexahedron:
    return mesh::CellType::quadrilateral;
  default:
    throw std::runtime_error("Unknown cell type.");
    return mesh::CellType::point;
  }
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
mesh::create_entities(mesh::CellType type, int dim)
{
  switch (type)
  {
  // case mesh::CellType::point:
  //   return create_entities_point(e, v);
  case mesh::CellType::interval:
    return create_entities_interval(dim);
  case mesh::CellType::triangle:
    return create_entities_triangle(dim);
  case mesh::CellType::tetrahedron:
    return create_entities_tetrahedron(dim);
  case mesh::CellType::quadrilateral:
    return create_entities_quadrilateral(dim);
  case mesh::CellType::hexahedron:
    return create_entities_hexahedron(dim);
  default:
    throw std::runtime_error("Unsupported cell type.");
    return Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();
  }
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
    return -1;
  }
}
//-----------------------------------------------------------------------------
int mesh::cell_num_entities(mesh::CellType type, int dim)
{
  assert(dim <= 3);
  static const int point[4] = {1, 0, 0, 0};
  static const int interval[4] = {2, 1, 0, 0};
  static const int triangle[4] = {3, 3, 1, 0};
  static const int quadrilateral[4] = {4, 4, 1, 0};
  static const int tetrahedron[4] = {4, 6, 4, 1};
  static const int hexahedron[4] = {8, 12, 6, 1};
  switch (type)
  {
  case mesh::CellType::point:
    return point[dim];
  case mesh::CellType::interval:
    return interval[dim];
  case mesh::CellType::triangle:
    return triangle[dim];
  case mesh::CellType::tetrahedron:
    return tetrahedron[dim];
  case mesh::CellType::quadrilateral:
    return quadrilateral[dim];
  case mesh::CellType::hexahedron:
    return hexahedron[dim];
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
