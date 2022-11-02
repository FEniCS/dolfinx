// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "cell_types.h"
#include <algorithm>
#include <basix/cell.h>
#include <cfloat>
#include <cstdlib>
#include <stdexcept>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::string mesh::to_string(CellType type)
{
  switch (type)
  {
  case CellType::point:
    return "point";
  case CellType::interval:
    return "interval";
  case CellType::triangle:
    return "triangle";
  case CellType::tetrahedron:
    return "tetrahedron";
  case CellType::quadrilateral:
    return "quadrilateral";
  case CellType::pyramid:
    return "pyramid";
  case CellType::prism:
    return "prism";
  case CellType::hexahedron:
    return "hexahedron";
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
mesh::CellType mesh::to_type(const std::string& cell)
{
  if (cell == "point")
    return CellType::point;
  else if (cell == "interval" or cell == "interval2D" or cell == "interval3D")
    return CellType::interval;
  else if (cell == "triangle" or cell == "triangle3D")
    return CellType::triangle;
  else if (cell == "tetrahedron")
    return CellType::tetrahedron;
  else if (cell == "quadrilateral" or cell == "quadrilateral3D")
    return CellType::quadrilateral;
  else if (cell == "pyramid")
    return CellType::pyramid;
  else if (cell == "prism")
    return CellType::prism;
  else if (cell == "hexahedron")
    return CellType::hexahedron;
  else
    throw std::runtime_error("Unknown cell type (" + cell + ")");
}
//-----------------------------------------------------------------------------
mesh::CellType mesh::cell_entity_type(CellType type, int d, int index)
{
  const int dim = cell_dim(type);
  if (d == dim)
    return type;
  else if (d == 1)
    return CellType::interval;
  else if (d == (dim - 1))
    return cell_facet_type(type, index);
  else
    return CellType::point;
}
//-----------------------------------------------------------------------------
mesh::CellType mesh::cell_facet_type(CellType type, int index)
{
  switch (type)
  {
  case CellType::point:
    return CellType::point;
  case CellType::interval:
    return CellType::point;
  case CellType::triangle:
    return CellType::interval;
  case CellType::tetrahedron:
    return CellType::triangle;
  case CellType::quadrilateral:
    return CellType::interval;
  case CellType::pyramid:
    throw std::runtime_error("TODO: pyramid");
  case CellType::prism:
    if (index == 0 or index == 4)
      return CellType::triangle;
    else
      return CellType::quadrilateral;
  case CellType::hexahedron:
    return CellType::quadrilateral;
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<int> mesh::get_entity_vertices(CellType type, int dim)
{
  const std::vector<std::vector<int>> topology
      = basix::cell::topology(cell_type_to_basix_type(type))[dim];
  return graph::AdjacencyList<int>(topology);
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<int> mesh::get_sub_entities(CellType type, int dim0,
                                                 int dim1)
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

  // TODO: get this data from basix
  static const std::vector<std::vector<int>> triangle = {{0, 1, 2}};
  static const std::vector<std::vector<int>> quadrilateral = {{0, 1, 2, 3}};
  static const std::vector<std::vector<int>> tetrahedron
      = {{0, 1, 2}, {0, 3, 4}, {1, 3, 5}, {2, 4, 5}};
  static const std::vector<std::vector<int>> prism
      = {{0, 1, 3}, {0, 2, 4, 6}, {1, 2, 5, 7}, {3, 4, 5, 8}, {6, 7, 8}};
  static const std::vector<std::vector<int>> hexahedron
      = {{0, 1, 3, 5},  {0, 2, 4, 8},  {1, 2, 6, 9},
         {3, 4, 7, 10}, {5, 6, 7, 11}, {8, 9, 10, 11}};
  switch (type)
  {
  case CellType::interval:
    return graph::AdjacencyList<int>(0);
  case CellType::point:
    return graph::AdjacencyList<int>(0);
  case CellType::triangle:
    return graph::AdjacencyList<int>(triangle);
  case CellType::tetrahedron:
    return graph::AdjacencyList<int>(tetrahedron);
  case CellType::prism:
    return graph::AdjacencyList<int>(prism);
  case CellType::quadrilateral:
    return graph::AdjacencyList<int>(quadrilateral);
  case CellType::hexahedron:
    return graph::AdjacencyList<int>(hexahedron);
  default:
    throw std::runtime_error("Unsupported cell type.");
  }
}
//-----------------------------------------------------------------------------
int mesh::cell_dim(CellType type)
{
  return basix::cell::topological_dimension(cell_type_to_basix_type(type));
}
//-----------------------------------------------------------------------------
int mesh::cell_num_entities(CellType type, int dim)
{
  assert(dim <= 3);
  constexpr std::array<int, 4> point = {1, 0, 0, 0};
  constexpr std::array<int, 4> interval = {2, 1, 0, 0};
  constexpr std::array<int, 4> triangle = {3, 3, 1, 0};
  constexpr std::array<int, 4> quadrilateral = {4, 4, 1, 0};
  constexpr std::array<int, 4> tetrahedron = {4, 6, 4, 1};
  constexpr std::array<int, 4> pyramid = {5, 8, 5, 1};
  constexpr std::array<int, 4> prism = {6, 9, 5, 1};
  constexpr std::array<int, 4> hexahedron = {8, 12, 6, 1};
  switch (type)
  {
  case CellType::point:
    return point[dim];
  case CellType::interval:
    return interval[dim];
  case CellType::triangle:
    return triangle[dim];
  case CellType::tetrahedron:
    return tetrahedron[dim];
  case CellType::quadrilateral:
    return quadrilateral[dim];
  case CellType::pyramid:
    return pyramid[dim];
  case CellType::prism:
    return prism[dim];
  case CellType::hexahedron:
    return hexahedron[dim];
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
bool mesh::is_simplex(CellType type) { return static_cast<int>(type) > 0; }
//-----------------------------------------------------------------------------
int mesh::num_cell_vertices(CellType type)
{
  return std::abs(static_cast<int>(type));
}
//-----------------------------------------------------------------------------
std::map<std::array<int, 2>, std::vector<std::set<int>>>
mesh::cell_entity_closure(CellType cell_type)
{
  const int cell_dim = mesh::cell_dim(cell_type);
  std::array<int, 4> num_entities;
  for (int i = 0; i <= cell_dim; ++i)
    num_entities[i] = cell_num_entities(cell_type, i);

  const graph::AdjacencyList<int> edge_v = get_entity_vertices(cell_type, 1);
  const auto face_e = get_sub_entities(cell_type, 2, 1);

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
        CellType face_type = cell_entity_type(cell_type, 2, entity);
        const int num_edges = cell_num_entities(face_type, 1);
        for (int e = 0; e < num_edges; ++e)
        {
          // Add edge
          const int edge_index = face_e.links(entity)[e];
          entity_closure[{{dim, entity}}][1].insert(edge_index);
          for (int v = 0; v < 2; ++v)
          {
            // Add vertex connected to edge
            entity_closure[{{dim, entity}}][0].insert(
                edge_v.links(edge_index)[v]);
          }
        }
      }

      if (dim == 1)
      {
        entity_closure[{{dim, entity}}][0].insert(edge_v.links(entity)[0]);
        entity_closure[{{dim, entity}}][0].insert(edge_v.links(entity)[1]);
      }
    }
  }

  return entity_closure;
}
//-----------------------------------------------------------------------------
basix::cell::type mesh::cell_type_to_basix_type(CellType celltype)
{
  switch (celltype)
  {
  case CellType::interval:
    return basix::cell::type::interval;
  case CellType::triangle:
    return basix::cell::type::triangle;
  case CellType::tetrahedron:
    return basix::cell::type::tetrahedron;
  case CellType::quadrilateral:
    return basix::cell::type::quadrilateral;
  case CellType::hexahedron:
    return basix::cell::type::hexahedron;
  case CellType::prism:
    return basix::cell::type::prism;
  case CellType::pyramid:
    return basix::cell::type::pyramid;
  default:
    throw std::runtime_error("Unrecognised cell type.");
  }
}
//-----------------------------------------------------------------------------
mesh::CellType mesh::cell_type_from_basix_type(basix::cell::type celltype)
{
  switch (celltype)
  {
  case basix::cell::type::interval:
    return CellType::interval;
  case basix::cell::type::triangle:
    return CellType::triangle;
  case basix::cell::type::tetrahedron:
    return CellType::tetrahedron;
  case basix::cell::type::quadrilateral:
    return CellType::quadrilateral;
  case basix::cell::type::hexahedron:
    return CellType::hexahedron;
  case basix::cell::type::prism:
    return CellType::prism;
  case basix::cell::type::pyramid:
    return CellType::pyramid;
  default:
    throw std::runtime_error("Unrecognised cell type.");
  }
}
//-----------------------------------------------------------------------------
