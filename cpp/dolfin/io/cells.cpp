// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "cells.h"
#include <dolfin/mesh/cell_types.h>
#include <numeric>
#include <stdexcept>

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<std::uint8_t> io::cells::dolfin_to_vtk(mesh::CellType type,
                                                   int num_nodes)
{
  switch (type)
  {
  case mesh::CellType::point:
    return {0};
  case mesh::CellType::interval:
    return {0, 1};
  case mesh::CellType::triangle:
  {
    std::vector<std::uint8_t> permutation(num_nodes);

    // Vertices
    int j = 0;
    permutation[j++] = 0;
    permutation[j++] = 1;
    permutation[j++] = 2;

    const int degree = mesh::cell_degree(type, num_nodes);
    for (int k = 1; k < degree; ++k)
      permutation[j++] = 3 + 2 * (degree - 1) + k - 1;
    for (int k = 1; k < degree; ++k)
      permutation[j++] = 3 + k - 1;
    for (int k = 1; k < degree; ++k)
      permutation[j++] = 2 * degree - (k - 1);

    // Interior VTK is ordered as a lower order triangle, while FEniCS
    // orders them lexicographically.
    // FIXME: Should be possible to generalize with some recursive function
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
      return permutation;
    }

    for (std::size_t k = 0; k < remainders.size(); ++k)
      permutation[j++] = base + remainders[k];
    return permutation;
  }
  case mesh::CellType::tetrahedron:
    switch (num_nodes)
    {
    case 4:
      return {0, 1, 2, 3};
    default:
      throw std::runtime_error("Higher order tetrahedron not supported.");
    }
  case mesh::CellType::quadrilateral:
  {
    // Check that num_nodes is a square integer (since quadrilaterals
    // are tensorproducts of intervals, the number of nodes for each
    // interval should be an integer)
    assert((std::sqrt(num_nodes) - std::floor(std::sqrt(num_nodes))) == 0);
    // Number of nodes in each direction
    const int n = sqrt(num_nodes);
    std::vector<std::uint8_t> permutation(num_nodes);

    // Vertices
    int j = 0;
    permutation[j++] = 0;
    permutation[j++] = n;
    permutation[j++] = n + 1;
    permutation[j++] = 1;

    // Edges
    for (int k = 2; k < n; ++k)
      permutation[j++] = n * k;
    for (int k = n + 2; k < 2 * n; ++k)
      permutation[j++] = k;
    for (int k = 2; k < n; ++k)
      permutation[j++] = k * n + 1;
    for (int k = 2; k < n; ++k)
      permutation[j++] = k;

    // Face
    for (int k = 2; k < n; ++k)
      for (int l = 2; l < n; ++l)
        permutation[j++] = l * n + k;

    assert(j == num_nodes);
    return permutation;
  }
  case mesh::CellType::hexahedron:
  {
    switch (num_nodes)
    {
    case 8:
      return {0, 1, 3, 2, 4, 5, 7, 6};
    default:
      throw std::runtime_error("Higher order hexahedron not supported.");
    }
  default:
    throw std::runtime_error("Unknown cell type.");
  }
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> io::cells::vtk_to_tp(mesh::CellType type,
                                               int num_nodes)
{
  const std::vector<std::uint8_t> reversed
      = io::cells::dolfin_to_vtk(type, num_nodes);
  switch (type)
  {
  case mesh::CellType::quadrilateral:
  {
    std::vector<std::uint8_t> perm(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
      perm[reversed[i]] = i;
    return perm;
  }
  case mesh::CellType::hexahedron:
  {
    std::vector<std::uint8_t> perm(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
      perm[reversed[i]] = i;
    return perm;
  }
  default:
    throw std::runtime_error("Simplicies can be expressed as TensorProduct");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> io::cells::lex_to_tp(mesh::CellType type,
                                               int num_nodes)
{
  switch (type)
  {
  case mesh::CellType::quadrilateral:
  {
    assert((std::sqrt(num_nodes) - std::floor(std::sqrt(num_nodes))) == 0);
    // Number of nodes in each direction
    const int n = sqrt(num_nodes);

    std::vector<std::uint8_t> permutation(num_nodes);
    std::vector<std::uint8_t> rows(n);
    std::iota(std::next(rows.begin()), std::prev(rows.end()), 2);
    rows.front() = 0;
    rows.back() = 1;

    int j = 0;
    for (auto row : rows)
    {
      permutation[j] = row;
      permutation[j + n - 1] = n + row;
      j++;
      for (int index = 0; index < n - 2; ++index)
      {
        permutation[j] = (2 + index) * n + row;
        j++;
      }
      j++;
    }
    return permutation;
  }
  case mesh::CellType::hexahedron:
  {
    switch (num_nodes)
    {
    case 8:
      return {0, 1, 3, 2, 4, 5, 7, 6};
    default:
      throw std::runtime_error("Higher order hexahedron not supported.");
    }
  }
  default:
    throw std::runtime_error("Simplicies can be expressed as TensorProduct.");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> io::cells::vtk_to_dolfin(mesh::CellType type,
                                                   int num_nodes)
{
  switch (type)
  {
  case mesh::CellType::point:
    return {0};
  case mesh::CellType::interval:
    return {0, 1};
  case mesh::CellType::triangle:
  {
    const std::vector<std::uint8_t> reversed
        = io::cells::dolfin_to_vtk(type, num_nodes);
    std::vector<std::uint8_t> perm(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
      perm[reversed[i]] = i;
    return perm;
  }
  case mesh::CellType::tetrahedron:
    switch (num_nodes)
    {
    case 4:
      return {0, 1, 2, 3};
    default:
      throw std::runtime_error("Higher order tetrahedron not supported.");
    }
  case mesh::CellType::quadrilateral:
    return io::cells::vtk_to_tp(type, num_nodes);
  case mesh::CellType::hexahedron:
    return io::cells::vtk_to_tp(type, num_nodes);
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
io::cells::gmsh_to_dolfin_ordering(
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    mesh::CellType type)
{
  /// Get VTK permutation for given cell type
  const std::vector<std::uint8_t> permutation
      = io::cells::vtk_to_dolfin(type, cells.cols());

  /// Permute input cells
  Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cells_dolfin(cells.rows(), cells.cols());
  for (Eigen::Index c = 0; c < cells_dolfin.rows(); ++c)
  {
    for (Eigen::Index v = 0; v < cells_dolfin.cols(); ++v)
      cells_dolfin(c, v) = cells(c, permutation[v]);
  }
  return cells_dolfin;
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t>
io::cells::default_cell_permutation(mesh::CellType type, int degree)
{
  switch (type)
  {
  case mesh::CellType::quadrilateral:
  {
    // Quadrilateral cells follow lexciographic order (LG) and must be
    // mapped to tensor product ordering.
    const int n = (degree + 1) * (degree + 1);
    switch (degree)
    {
    case 1:
      // Current default for built in meshes
      return io::cells::lex_to_tp(type, n);
    default:
      // mesh::compute_local_to_global_point_map assumes that the first
      // four points in the connectivity array are the vertices, thus
      // you need VTK ordering.
      return io::cells::dolfin_to_vtk(type, n);
    }
  }
  case mesh::CellType::hexahedron:
  {
    switch (degree)
    {
    case 1:
      // First order hexes follows lexiographic ordering
      return {0, 1, 2, 3, 4, 5, 6, 7};
    default:
      throw std::runtime_error("Higher order hexahedron not supported");
    }
    break;
  }
  case mesh::CellType::point:
    return io::cells::dolfin_to_vtk(type, 1);
  case mesh::CellType::interval:
    return io::cells::dolfin_to_vtk(type, 2);
  case mesh::CellType::tetrahedron:
    return io::cells::dolfin_to_vtk(type, 4);
  case mesh::CellType::triangle:
  {
    const int n = (degree + 1) * (degree + 2) / 2;
    return io::cells::dolfin_to_vtk(type, n);
  }
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
