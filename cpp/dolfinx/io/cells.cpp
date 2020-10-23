// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "cells.h"
#include <dolfinx/common/log.h>
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
      return 1;
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
  map[1] = n;
  map[2] = n + 1;
  map[3] = 1;

  int j = 4;

  // Edges
  for (int k = 2; k < n; ++k)
    map[j++] = n * k;
  for (int k = n + 2; k < 2 * n; ++k)
    map[j++] = k;
  for (int k = 2; k < n; ++k)
    map[j++] = k * n + 1;
  for (int k = 2; k < n; ++k)
    map[j++] = k;

  // Face
  for (int k = 2; k < n; ++k)
    for (int l = 2; l < n; ++l)
      map[j++] = l * n + k;
  assert(j == num_nodes);

  return map;
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> vtk_hexahedron(int num_nodes)
{
  switch (num_nodes)
  {
  case 8:
    return {0, 4, 6, 2, 1, 5, 7, 3};
  case 27:
    // This is the documented VTK ordering
    return {0,  9, 12, 3,  1,  10, 13, 4,  18, 15, 21, 6,  19, 16,
            22, 7, 2,  11, 14, 5,  8,  17, 20, 23, 24, 25, 26};
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
    throw std::runtime_error("Higher order GMSH triangle not supported");
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
    throw std::runtime_error("Higher order GMSH tetrahedron not supported");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> gmsh_hexahedron(int num_nodes)
{
  switch (num_nodes)
  {
  case 8:
    return {0, 4, 6, 2, 1, 5, 7, 3};
  case 27:
    return {0,  9, 12, 3, 1,  10, 13, 4,  18, 6,  2,  15, 11, 21,
            14, 5, 19, 7, 16, 22, 24, 20, 8,  17, 23, 25, 26};
  default:
    throw std::runtime_error("Higher order GMSH hexahedron not supported");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> gmsh_quadrilateral(int num_nodes)
{
  switch (num_nodes)
  {
  case 4:
    return {0, 2, 3, 1};
  case 9:
    return {0, 3, 4, 1, 6, 5, 7, 2, 8};
  case 16:
    return {0, 4, 5, 1, 8, 12, 6, 7, 13, 9, 3, 2, 10, 14, 15, 11};
  default:
    throw std::runtime_error("Higher order GMSH quadrilateral not supported");
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
Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
io::cells::compute_permutation(
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    const std::vector<std::uint8_t>& p)
{
  Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cells_new(cells.rows(), cells.cols());
  for (Eigen::Index c = 0; c < cells_new.rows(); ++c)
  {
    auto cell = cells.row(c);
    auto cell_new = cells_new.row(c);
    for (Eigen::Index i = 0; i < cell_new.size(); ++i)
      cell_new(i) = cell(p[i]);
  }
  return cells_new;
}
//-----------------------------------------------------------------------------
