// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "cell_types.h"
#include "Geometry.h"
#include "MeshEntity.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <dolfin/common/log.h>
#include <stdexcept>

using namespace dolfin;

namespace
{
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
get_entity_vertices_interval(int dim)
{
  const static Eigen::Array<int, 2, 1> e0
      = (Eigen::Array<int, 2, 1>() << 0, 1).finished();
  const static Eigen::Array<int, 1, 2, Eigen::RowMajor> e1
      = (Eigen::Array<int, 1, 2, Eigen::RowMajor>() << 0, 1).finished();
  switch (dim)
  {
  case 0:
    return e0;
  case 1:
    return e1;
  default:
    return Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();
  }
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
get_entity_vertices_triangle(int dim)
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
get_entity_vertices_quadrilateral(int dim)
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
get_entity_vertices_tetrahedron(int dim)
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
get_entity_vertices_hexahedron(int dim)
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
mesh::get_entity_vertices(mesh::CellType type, int dim)
{
  switch (type)
  {
  // case mesh::CellType::point:
  //   return create_entities_point(e, v);
  case mesh::CellType::interval:
    return get_entity_vertices_interval(dim);
  case mesh::CellType::triangle:
    return get_entity_vertices_triangle(dim);
  case mesh::CellType::tetrahedron:
    return get_entity_vertices_tetrahedron(dim);
  case mesh::CellType::quadrilateral:
    return get_entity_vertices_quadrilateral(dim);
  case mesh::CellType::hexahedron:
    return get_entity_vertices_hexahedron(dim);
  default:
    throw std::runtime_error("Unsupported cell type.");
    return Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();
  }
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
mesh::get_sub_entities(CellType type, int dim0, int dim1)
{
  if (dim0 != 2)
  {
    throw std::runtime_error(
        "mesh::get_sub_entities supports faces (d=2) only at present.");
  }
  if (dim1 != 1)
  {
    throw std::runtime_error(
        "mesh::get_sub_entities supports getting edges (d=1) at present.");
  }
  const static Eigen::Array<int, 1, 3, Eigen::RowMajor> triangle
      = (Eigen::Array<int, 1, 3, Eigen::RowMajor>() << 0, 1, 2).finished();
  const static Eigen::Array<int, 1, 4, Eigen::RowMajor> quadrilateral
      = (Eigen::Array<int, 1, 4, Eigen::RowMajor>() << 0, 3, 1, 2).finished();
  const static Eigen::Array<int, 4, 3, Eigen::RowMajor> tetrahedron
      = (Eigen::Array<int, 4, 3, Eigen::RowMajor>() << 0, 1, 2, 0, 3, 4, 1, 3,
         5, 2, 4, 5)
            .finished();
  const static Eigen::Array<int, 6, 4, Eigen::RowMajor> hexahedron
      = (Eigen::Array<int, 6, 4, Eigen::RowMajor>() << 0, 1, 4, 5, 2, 3, 6, 7,
         0, 2, 8, 9, 1, 3, 10, 11, 4, 6, 8, 10, 5, 7, 9, 11)
            .finished();

  switch (type)
  {
  case mesh::CellType::interval:
    return Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();
  case mesh::CellType::point:
    return Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();
  case mesh::CellType::triangle:
    return triangle;
  case mesh::CellType::tetrahedron:
    return tetrahedron;
  case mesh::CellType::quadrilateral:
    return quadrilateral;
  case mesh::CellType::hexahedron:
    return hexahedron;
  default:
    throw std::runtime_error("Unsupported cell type.");
    return Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();
  }

  // static const int triangle[][4] = {
  //     {0, 1, 2, -1},
  // };
  // static const int tetrahedron[][4]
  //     = {{0, 1, 2, -1}, {0, 3, 4, -1}, {1, 3, 5, -1}, {2, 4, 5, -1}};
  // static const int quadrilateral[][4] = {{0, 3, 1, 2}};
  // static const int hexahedron[][4]
  //     = {{0, 1, 4, 5},   {2, 3, 6, 7},  {0, 2, 8, 9},
  //        {1, 3, 10, 11}, {4, 6, 8, 10}, {5, 7, 9, 11}};
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
std::map<std::array<int, 2>, std::vector<std::set<int>>>
mesh::cell_entity_closure(mesh::CellType cell_type)
{
  const int cell_dim = mesh::cell_dim(cell_type);
  std::array<int, 4> num_entities{};
  for (int i = 0; i <= cell_dim; ++i)
    num_entities[i] = mesh::cell_num_entities(cell_type, i);

  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      edge_v = mesh::get_entity_vertices(cell_type, 1);
  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      face_e = mesh::get_sub_entities(cell_type, 2, 1);

  std::map<std::array<int, 2>, std::vector<std::set<int>>> entity_closure;
  for (int dim = 0; dim <= cell_dim; ++dim)
  {
    for (int entity = 0; entity < num_entities[dim]; ++entity)
    {
      // Add self
      entity_closure[{{dim, entity}}].resize(cell_dim + 1);
      entity_closure[{{dim, entity}}][dim].insert(entity);

      if (dim == 3)
      {
        // Add all sub-entities
        for (int f = 0; f < num_entities[2]; ++f)
          entity_closure[{{dim, entity}}][2].insert(f);
        for (int e = 0; e < num_entities[1]; ++e)
          entity_closure[{{dim, entity}}][1].insert(e);
        for (int v = 0; v < num_entities[0]; ++v)
          entity_closure[{{dim, entity}}][0].insert(v);
      }

      if (dim == 2)
      {
        mesh::CellType face_type = mesh::cell_entity_type(cell_type, 2);
        const int num_edges = mesh::cell_num_entities(face_type, 1);
        for (int e = 0; e < num_edges; ++e)
        {
          // Add edge
          const int edge_index = face_e(entity, e);
          entity_closure[{{dim, entity}}][1].insert(edge_index);
          for (int v = 0; v < 2; ++v)
          {
            // Add vertex connected to edge
            entity_closure[{{dim, entity}}][0].insert(edge_v(edge_index, v));
          }
        }
      }

      if (dim == 1)
      {
        entity_closure[{{dim, entity}}][0].insert(edge_v(entity, 0));
        entity_closure[{{dim, entity}}][0].insert(edge_v(entity, 1));
      }
    }
  }

  return entity_closure;
}
//-----------------------------------------------------------------------------
int mesh::cell_degree(mesh::CellType type, int num_nodes)
{
  switch (type)
  {
  case mesh::CellType::point:
    return 1;
  case mesh::CellType::interval:
    return 1;
  case mesh::CellType::triangle:
    switch (num_nodes)
    {
    case 3:
      return 1;
    case 6:
      return 2;
    case 10:
      return 3;
    case 15:
      return 4;
    case 21:
      return 5;
    case 28:
      return 6;
    case 36:
      return 7;
    case 45:
      LOG(WARNING) << "8th order mesh is untested";
      return 8;
    case 55:
      LOG(WARNING) << "9th order mesh is untested";
      return 9;
    default:
      throw std::runtime_error("Unknown triangle layout.");
    }
  case mesh::CellType::tetrahedron:
    switch (num_nodes)
    {
    case 4:
      return 1;
    case 10:
      return 2;
    case 20:
      return 3;
    default:
      throw std::runtime_error("Unknown tetrahedron layout.");
    }
  case mesh::CellType::quadrilateral:
  {
    const int n = std::sqrt(num_nodes);
    if (num_nodes != n * n)
    {
      throw std::runtime_error("Quadrilateral of order "
                               + std::to_string(num_nodes) + " not supported");
    }
    return n - 1;
  }
  case mesh::CellType::hexahedron:
    switch (num_nodes)
    {
    case 8:
      return 1;
    case 27:
      return 2;
    default:
      throw std::runtime_error("Unsupported hexahedron layout");
      return 1;
    }
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}

std::vector<int> mesh::cell_vertex_index(mesh::CellType type, int num_nodes,
                                         int num_vertices_per_cell)
{
  int degree = mesh::cell_degree(type, num_nodes);
  switch (type)
  {
  case mesh::CellType::quadrilateral:
    // Topographical ordering yields this
    return {0, 1, degree + 1, degree + 2};
  case mesh::CellType::hexahedron:
    if (num_nodes == 8)
    {
      std::vector<int> vertex_indices(num_vertices_per_cell);
      std::iota(vertex_indices.begin(), vertex_indices.end(), 0);
      return vertex_indices;
    }
    else
    {
      const int spacing = (1 + degree) * (1 + degree);
      return {0,       1,           degree + 1,           degree + 2,
              spacing, spacing + 1, spacing + degree + 1, spacing + degree + 2};
    }
  default:
    std::vector<int> vertex_indices(num_vertices_per_cell);
    std::iota(vertex_indices.begin(), vertex_indices.end(), 0);
    return vertex_indices;
  }
}
//-----------------------------------------------------------------------------
