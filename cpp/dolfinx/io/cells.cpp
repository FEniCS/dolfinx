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

int io::cells::cell_degree(mesh::CellType type, int num_nodes)
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
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::vector<std::uint8_t> io::cells::vtk_to_dolfin(mesh::CellType type,
                                                   int num_nodes)
{
  switch (type)
  {
  case mesh::CellType::point:
    return {0};
  case mesh::CellType::interval:
  {
    std::vector<std::uint8_t> permutation(num_nodes);
    std::iota(permutation.begin(), permutation.end(), 0);
    return permutation;
  }
  case mesh::CellType::triangle:
  {
    std::vector<std::uint8_t> permutation(num_nodes);

    // Vertices
    permutation[0] = 0;
    permutation[1] = 1;
    permutation[2] = 2;

    int j = 3;
    const int degree = cell_degree(type, num_nodes);
    for (int k = 1; k < degree; ++k)
      permutation[j++] = 3 + 2 * (degree - 1) + k - 1;
    for (int k = 1; k < degree; ++k)
      permutation[j++] = 3 + k - 1;
    for (int k = 1; k < degree; ++k)
      permutation[j++] = 2 * degree - (k - 1);

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
    case 10:
      return {0, 1, 2, 3, 9, 6, 8, 7, 5, 4};
    case 20:
      return {0,  1,  2, 3, 14, 15, 8,  9,  13, 12,
              10, 11, 6, 7, 4,  5,  18, 16, 17, 19};
    default:
      throw std::runtime_error("Unknown tetrahedron layout");
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
    permutation[0] = 0;
    permutation[1] = n;
    permutation[2] = n + 1;
    permutation[3] = 1;

    int j = 4;

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
      return {0, 4, 6, 2, 1, 5, 7, 3};
    case 27:
      return {0,  9, 12, 3,  1,  10, 13, 4,  18, 15, 21, 6,  19, 16,
              22, 7, 2,  11, 14, 5,  8,  17, 20, 23, 24, 25, 26};
    default:
      throw std::runtime_error("Higher order hexahedron not supported.");
    }
  default:
    throw std::runtime_error("Unknown cell type.");
  }
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> io::cells::dolfin_to_vtk(mesh::CellType type,
                                                   int num_nodes)
{
  const std::vector<std::uint8_t> reversed
      = io::cells::vtk_to_dolfin(type, num_nodes);
  std::vector<std::uint8_t> perm(num_nodes);
  for (int i = 0; i < num_nodes; ++i)
    perm[reversed[i]] = i;
  return perm;
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> io::cells::gmsh_to_dolfin(std::string type)
{

  if (type == "tetra")
  {
    return std::vector<std::uint8_t>{0, 1, 2, 3};
  }
  else if (type == "tetra10")
  {
    // NOTE: GMSH DOCUMENTATION IS WRONG, it would have the following
    // permutation:
    // return std::vector<std::uint8_t>{0, 1, 2, 3, 9, 6, 8, 7, 4, 5};
    // This is the one returned by pygmsh
    return std::vector<std::uint8_t>{0, 1, 2, 3, 9, 6, 8, 7, 5, 4};
  }
  else if (type == "tetra20")
  {
    return std::vector<std::uint8_t>{0,  1,  2, 3, 14, 15, 8,  9,  13, 12,
                                     11, 10, 5, 4, 7,  6,  19, 18, 17, 16};
  }
  else if (type == "hexahedron")
    return std::vector<std::uint8_t>{0, 4, 6, 2, 1, 5, 7, 3};
  else if (type == "hexahedron27")
  {
    return std::vector<std::uint8_t>{0,  9,  12, 3,  1,  10, 13, 4,  18,
                                     6,  2,  15, 11, 21, 14, 5,  19, 7,
                                     16, 22, 24, 20, 8,  17, 23, 25, 26};
  }
  else if (type == "triangle")
    return std::vector<std::uint8_t>{0, 1, 2};
  else if (type == "triangle6")
    return std::vector<std::uint8_t>{0, 1, 2, 5, 3, 4};
  else if (type == "triangle10")
    return std::vector<std::uint8_t>{0, 1, 2, 7, 8, 3, 4, 6, 5, 9};
  else if (type == "quad")
    return std::vector<std::uint8_t>{0, 2, 3, 1};
  else if (type == "quad9")
    return std::vector<std::uint8_t>{0, 3, 4, 1, 6, 5, 7, 2, 8};
  else if (type == "quad16")
  {
    return std::vector<std::uint8_t>{0,  4, 5, 1, 8,  12, 6,  7,
                                     13, 9, 3, 2, 10, 14, 15, 11};
  }
  else
    throw std::runtime_error("Gmsh cell type not recognized");
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> io::cells::dolfin_to_gmsh(std::string type)
{
  const std::vector<std::uint8_t> reversed = io::cells::gmsh_to_dolfin(type);
  std::vector<std::uint8_t> perm(reversed.size());
  for (std::size_t i = 0; i < perm.size(); ++i)
    perm[reversed[i]] = i;
  return perm;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
io::cells::permute_ordering(
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    const std::vector<std::uint8_t>& permutation)
{
  Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cells_new(cells.rows(), cells.cols());
  for (Eigen::Index c = 0; c < cells_new.rows(); ++c)
  {
    for (Eigen::Index v = 0; v < cells_new.cols(); ++v)
      cells_new(c, permutation[v]) = cells(c, v);
  }
  return cells_new;
}
//-----------------------------------------------------------------------------
