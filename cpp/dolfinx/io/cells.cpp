// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "cells.h"
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <numeric>
#include <stdexcept>

using namespace dolfinx;
namespace
{
int cell_degree(mesh::CellType type, int num_nodes)
{
  switch (type)
  {
  case mesh::CellType::point:
    return 1;
  case mesh::CellType::interval:
    return num_nodes - 1;
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
      throw std::runtime_error("Unknown triangle layout. Number of nodes: "
                               + std::to_string(num_nodes));
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
    }
  case mesh::CellType::prism:
    switch (num_nodes)
    {
    case 6:
      return 1;
    case 15:
      return 2;
    default:
      throw std::runtime_error("Unsupported prism layout");
    }
  case mesh::CellType::pyramid:
    switch (num_nodes)
    {
    case 5:
      return 1;
    case 13:
      return 2;
    default:
      throw std::runtime_error("Unsupported pyramid layout");
    }
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> vtk_triangle(int num_nodes)
{
  // Vertices
  std::vector<std::uint8_t> map(num_nodes);
  std::iota(map.begin(), map.begin() + 3, 0);

  int j = 3;
  const int degree = cell_degree(mesh::CellType::triangle, num_nodes);
  for (int k = 1; k < degree; ++k)
    map[j++] = 3 + 2 * (degree - 1) + k - 1;
  for (int k = 1; k < degree; ++k)
    map[j++] = 3 + k - 1;
  for (int k = 1; k < degree; ++k)
    map[j++] = 2 * degree - (k - 1);

  if (degree < 3)
    return map;

  // Interior VTK is ordered as a lower order triangle, while FEniCS
  // orders them lexicographically.
  // FIXME: Should be possible to generalize with some recursive
  //        function
  std::vector<std::uint8_t> remainders(num_nodes - j);
  const int base = 3 * degree;
  switch (degree)
  {
  case 3:
    remainders = {0};
    break;
  case 4:
    remainders = {0, 1, 2};
    break;
  case 5:
    remainders = {0, 2, 5, 1, 4, 3};
    break;
  case 6:
    remainders = {0, 3, 9, 1, 2, 6, 8, 7, 4, 5};
    break;
  case 7:
    remainders = {0, 4, 14, 1, 2, 3, 8, 11, 13, 12, 9, 5, 6, 7, 10};
    break;
  case 8:
    remainders = {0,  5,  20, 1, 2, 3, 4,  10, 14, 17, 19,
                  18, 15, 11, 6, 7, 9, 16, 8,  13, 12};
    break;
  case 9:
    remainders = {0,  6,  27, 1, 2, 3,  4,  5, 12, 17, 21, 24, 26, 25,
                  22, 18, 13, 7, 8, 11, 23, 9, 10, 16, 20, 19, 14, 15};
    break;
  default:
    throw std::runtime_error("Unknown triangle layout: "
                             + std::to_string(degree));
  }

  for (std::size_t k = 0; k < remainders.size(); ++k)
    map[j + k] = base + remainders[k];

  return map;
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> vtk_tetrahedron(int num_nodes)
{
  switch (num_nodes)
  {
  case 4:
    return {0, 1, 2, 3};
  case 10:
    return {0, 1, 2, 3, 9, 6, 8, 7, 5, 4};
  case 20:
    return {0,  1,  2, 3, 14, 15, 8,  9,  13, 12,
            10, 11, 6, 7, 4,  5,  18, 16, 17, 19};
  default:
    throw std::runtime_error("Unknown tetrahedron layout");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> vtk_wedge(int num_nodes)
{
  switch (num_nodes)
  {
  case 6:
    return {0, 1, 2, 3, 4, 5};
  case 15:
    return {0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11};
  default:
    throw std::runtime_error("Unknown wedge layout");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> vtk_pyramid(int num_nodes)
{
  switch (num_nodes)
  {
  case 5:
    return {0, 1, 3, 2, 4};
  case 13:
    return {0, 1, 3, 2, 4, 5, 8, 10, 6, 7, 9, 12, 11};
  default:
    throw std::runtime_error("Unknown pyramid layout");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> vtk_quadrilateral(int num_nodes)
{
  // Check that num_nodes is a square integer (since quadrilaterals are
  // tensorproducts of intervals, the number of nodes for each interval
  // should be an integer)
  assert((std::sqrt(num_nodes) - std::floor(std::sqrt(num_nodes))) == 0);

  // Number of nodes in each direction
  const int n = sqrt(num_nodes);
  std::vector<std::uint8_t> map(num_nodes);

  // Vertices
  map[0] = 0;
  map[1] = 1;
  map[2] = 3;
  map[3] = 2;

  int j = 4;

  const int edge_nodes = n - 2;

  // Edges
  for (int k = 0; k < edge_nodes; ++k)
    map[j++] = 4 + k;
  for (int k = 0; k < edge_nodes; ++k)
    map[j++] = 4 + 2 * edge_nodes + k;
  for (int k = 0; k < edge_nodes; ++k)
    map[j++] = 4 + 3 * edge_nodes + k;
  for (int k = 0; k < edge_nodes; ++k)
    map[j++] = 4 + edge_nodes + k;

  // Face
  for (int k = 0; k < edge_nodes * edge_nodes; ++k)
    map[j++] = 4 + edge_nodes * 4 + k;
  return map;
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> vtk_hexahedron(int num_nodes)
{
  switch (num_nodes)
  {
  case 8:
    return {0, 1, 3, 2, 4, 5, 7, 6};
  case 27:
    // This is the documented VTK ordering
    return {0,  1,  3,  2,  4,  5,  7,  6,  8,  11, 13, 9,  16, 18,
            19, 17, 10, 12, 15, 14, 22, 23, 21, 24, 20, 25, 26};
  default:
    throw std::runtime_error("Higher order hexahedron not supported.");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> gmsh_triangle(int num_nodes)
{
  switch (num_nodes)
  {
  case 3:
    return {0, 1, 2};
  case 6:
    return {0, 1, 2, 5, 3, 4};
  case 10:
    return {0, 1, 2, 7, 8, 3, 4, 6, 5, 9};
  default:
    throw std::runtime_error("Higher order Gmsh triangle not supported");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> gmsh_tetrahedron(int num_nodes)
{
  switch (num_nodes)
  {
  case 4:
    return {0, 1, 2, 3};
  case 10:
    return {0, 1, 2, 3, 9, 6, 8, 7, 4, 5};
  case 20:
    return {0,  1,  2, 3, 14, 15, 8,  9,  13, 12,
            11, 10, 5, 4, 7,  6,  19, 18, 17, 16};
  default:
    throw std::runtime_error("Higher order Gmsh tetrahedron not supported");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> gmsh_hexahedron(int num_nodes)
{
  switch (num_nodes)
  {
  case 8:
    return {0, 1, 3, 2, 4, 5, 7, 6};
  case 27:
    return {0,  1,  3,  2,  4,  5,  7,  6,  8,  9,  10, 11, 12, 13,
            15, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
  default:
    throw std::runtime_error("Higher order Gmsh hexahedron not supported");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> gmsh_quadrilateral(int num_nodes)
{
  switch (num_nodes)
  {
  case 4:
    return {0, 1, 3, 2};
  case 9:
    return {0, 1, 3, 2, 4, 6, 7, 5, 8};
  case 16:
    return {0, 1, 3, 2, 4, 5, 8, 9, 11, 10, 7, 6, 12, 13, 15, 14};
  default:
    throw std::runtime_error("Higher order Gmsh quadrilateral not supported");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> gmsh_prism(int num_nodes)
{
  switch (num_nodes)
  {
  case 6:
    return {0, 1, 2, 3, 4, 5};
  case 15:
    return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  default:
    throw std::runtime_error("Higher order Gmsh prism not supported");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> gmsh_pyramid(int num_nodes)
{
  switch (num_nodes)
  {
  case 5:
    return {0, 1, 3, 2, 4};
  case 13:
    return {0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 12, 11};
  default:
    throw std::runtime_error("Higher order Gmsh pyramid not supported");
  }
}
} // namespace
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> io::cells::perm_vtk(mesh::CellType type,
                                              int num_nodes)
{
  std::vector<std::uint8_t> map;
  switch (type)
  {
  case mesh::CellType::point:
    map = {0};
    break;
  case mesh::CellType::interval:
    map.resize(num_nodes);
    std::iota(map.begin(), map.end(), 0);
    break;
  case mesh::CellType::triangle:
    map = vtk_triangle(num_nodes);
    break;
  case mesh::CellType::tetrahedron:
    map = vtk_tetrahedron(num_nodes);
    break;
  case mesh::CellType::quadrilateral:
    map = vtk_quadrilateral(num_nodes);
    break;
  case mesh::CellType::hexahedron:
    map = vtk_hexahedron(num_nodes);
    break;
  case mesh::CellType::prism:
    map = vtk_wedge(num_nodes);
    break;
  case mesh::CellType::pyramid:
    map = vtk_pyramid(num_nodes);
    break;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return io::cells::transpose(map);
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> io::cells::perm_gmsh(const mesh::CellType type,
                                               const int num_nodes)
{
  std::vector<std::uint8_t> map;
  switch (type)
  {
  case mesh::CellType::point:
    map = {0};
    break;
  case mesh::CellType::interval:
    map.resize(num_nodes);
    std::iota(map.begin(), map.end(), 0);
    break;
  case mesh::CellType::triangle:
    map = gmsh_triangle(num_nodes);
    break;
  case mesh::CellType::tetrahedron:
    map = gmsh_tetrahedron(num_nodes);
    break;
  case mesh::CellType::quadrilateral:
    map = gmsh_quadrilateral(num_nodes);
    break;
  case mesh::CellType::hexahedron:
    map = gmsh_hexahedron(num_nodes);
    break;
  case mesh::CellType::prism:
    map = gmsh_prism(num_nodes);
    break;
  case mesh::CellType::pyramid:
    map = gmsh_pyramid(num_nodes);
    break;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return io::cells::transpose(map);
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t>
io::cells::transpose(const std::vector<std::uint8_t>& map)
{
  std::vector<std::uint8_t> transpose(map.size());
  for (std::size_t i = 0; i < map.size(); ++i)
    transpose[map[i]] = i;
  return transpose;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
io::cells::apply_permutation(const std::span<const std::int64_t>& cells,
                             std::array<std::size_t, 2> shape,
                             const std::span<const std::uint8_t>& p)
{
  assert(cells.size() == shape[0] * shape[1]);
  assert(shape[1] == p.size());

  LOG(INFO) << "IO permuting cells";
  std::vector<std::int64_t> cells_new(cells.size());
  for (std::size_t c = 0; c < shape[0]; ++c)
  {
    auto cell = cells.subspan(c * shape[1], shape[1]);
    std::span cell_new(cells_new.data() + c * shape[1], shape[1]);
    for (std::size_t i = 0; i < shape[1]; ++i)
      cell_new[i] = cell[p[i]];
  }
  return cells_new;
}
//-----------------------------------------------------------------------------
std::int8_t io::cells::get_vtk_cell_type(mesh::CellType cell, int dim)
{
  if (cell == mesh::CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cell");

  // Get cell type
  mesh::CellType cell_type = mesh::cell_entity_type(cell, dim, 0);

  // Determine VTK cell type (arbitrary Lagrange elements)
  // https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
  switch (cell_type)
  {
  case mesh::CellType::point:
    return 1;
  case mesh::CellType::interval:
    return 68;
  case mesh::CellType::triangle:
    return 69;
  case mesh::CellType::quadrilateral:
    return 70;
  case mesh::CellType::tetrahedron:
    return 71;
  case mesh::CellType::hexahedron:
    return 72;
  default:
    throw std::runtime_error("Unknown cell type");
  }
}
//----------------------------------------------------------------------------
