// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "cells.h"
#include <dolfinx/mesh/cell_types.h>
#include <numeric>
#include <stdexcept>

using namespace dolfinx;

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
    const int degree = mesh::cell_degree(type, num_nodes);
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
      // // TODO: change permutation when paraview issue 19433 is resolved
      // // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
      // return {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
      //         22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};

      // This is the documented VTK ordering
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
  // Transpose of vtk_to_dolfin
  const std::vector<std::uint8_t> perm_r
      = io::cells::vtk_to_dolfin(type, num_nodes);
  assert(num_nodes == (int)perm_r.size());
  std::vector<std::uint8_t> perm(perm_r.size());
  for (std::size_t i = 0; i < perm.size(); ++i)
    perm[perm_r[i]] = i;
  return perm;
}
//-----------------------------------------------------------------------------
// std::vector<std::uint8_t> io::cells::lex_to_tp(mesh::CellType type,
//                                                int num_nodes)
// {
//   switch (type)
//   {
//   case mesh::CellType::quadrilateral:
//   {
//     assert((std::sqrt(num_nodes) - std::floor(std::sqrt(num_nodes))) == 0);
//     // Number of nodes in each direction
//     const int n = sqrt(num_nodes);

//     std::vector<std::uint8_t> permutation(num_nodes);
//     std::vector<std::uint8_t> rows(n);
//     std::iota(std::next(rows.begin()), std::prev(rows.end()), 2);
//     rows.front() = 0;
//     rows.back() = 1;

//     int j = 0;
//     for (auto row : rows)
//     {
//       permutation[j] = row;
//       permutation[j + n - 1] = n + row;
//       j++;
//       for (int index = 0; index < n - 2; ++index)
//       {
//         permutation[j] = (2 + index) * n + row;
//         j++;
//       }
//       j++;
//     }
//     return permutation;
//   }
//   case mesh::CellType::hexahedron:
//   {
//     switch (num_nodes)
//     {
//     case 8:
//       return {0, 4, 2, 6, 1, 5, 3, 7};
//     default:
//       throw std::runtime_error("Higher order hexahedron not supported.");
//     }
//   }
//   default:
//     throw std::runtime_error("Simplicies can be expressed as
//     TensorProduct.");
//   }
// }
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
