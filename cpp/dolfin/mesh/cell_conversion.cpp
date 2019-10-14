// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Geometry.h"
#include "MeshEntity.h"
#include "cell_types.h"
#include <Eigen/Dense>
#include <cfloat>
#include <cstdlib>
#include <dolfin/common/log.h>
#include <dolfin/mesh/cell_conversion.h>
#include <dolfin/mesh/cell_types.h>
#include <iostream>
#include <stdexcept>

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<std::uint8_t> mesh::vtk_cell_permutation(mesh::CellType type,
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

    int degree = mesh::cell_degree(type, num_nodes);

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
    default:
      return permutation;
    }
    for (int k = 0; k < unsigned(remainders.size()); ++k)
      permutation[j++] = base + remainders[k];
    return permutation;
  }
  case mesh::CellType::tetrahedron:
    switch (num_nodes)
    {
    case 4:
      return {0, 1, 2, 3};
    default:
      throw std::runtime_error("Higher order tetrahedron not supported");
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
    switch (num_nodes)
    {
    case 8:
      return {0, 1, 3, 2, 4, 5, 7, 6};
    default:
      throw std::runtime_error("Higher order hexahedron not supported");
    }
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> mesh::vtk_to_tp(mesh::CellType type, int num_nodes)
{
  switch (type)
  {
    {
    case mesh::CellType::quadrilateral:
    {
      std::vector<std::uint8_t> reversed
          = mesh::vtk_cell_permutation(type, num_nodes);
      std::vector<std::uint8_t> perm(num_nodes);
      for (int i = 0; i < num_nodes; ++i)
        perm[reversed[i]] = i;
      return perm;
    }
    case mesh::CellType::hexahedron:
      std::vector<std::uint8_t> reversed
          = mesh::vtk_cell_permutation(type, num_nodes);
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
std::vector<std::uint8_t> mesh::lexico_to_tp(mesh::CellType type, int num_nodes)
{
  switch (type)
  {
  case mesh::CellType::quadrilateral:
  {
    assert((std::sqrt(num_nodes) - std::floor(std::sqrt(num_nodes))) == 0);
    // Number of nodes in each direction
    const int n = sqrt(num_nodes);

    std::vector<std::uint8_t> permutation(num_nodes);
    int j = 0;
    std::vector<std::uint8_t> rows(n);
    std::iota(std::next(rows.begin()), std::prev(rows.end()), 2);
    rows.front() = 0;
    rows.back() = 1;

    std::vector<std::uint8_t>::iterator row;
    for (row = rows.begin(); row != rows.end(); ++row)
    {
      permutation[j] = *row;
      permutation[j + n - 1] = n + *row;
      j++;
      for (int index = 0; index < n - 2; ++index)
      {
        permutation[j] = (2 + index) * n + *row;
        j++;
      }
      j++;
    }
    return permutation;
  }
  case mesh::CellType::hexahedron:
    switch (num_nodes)
    {
    case 8:
      return {0, 1, 3, 2, 4, 5, 7, 6};
    default:
      throw std::runtime_error("Higher order hexahedron not supported");
    }
  default:
    throw std::runtime_error("Simplicies can be expressed as TensorProduct");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> mesh::vtk_to_fenics(mesh::CellType type,
                                              int num_nodes)
{
  switch (type)
  {
    {
    case mesh::CellType::point:
      return {0};
    case mesh::CellType::interval:
      return {0, 1};
    case mesh::CellType::triangle:
    {
      std::vector<std::uint8_t> reversed
          = mesh::vtk_cell_permutation(type, num_nodes);
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
        throw std::runtime_error("Higher order tetrahedron not supported");
      }
    case mesh::CellType::quadrilateral:
    {
      std::vector<std::uint8_t> perm = mesh::vtk_to_tp(type, num_nodes);
      return perm;
    }
    case mesh::CellType::hexahedron:
    {
      std::vector<std::uint8_t> perm = mesh::vtk_to_tp(type, num_nodes);
      return perm;
    }
    default:
      throw std::runtime_error("Unknown cell type.");
    }
  }
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
mesh::gmsh_to_dolfin_ordering(
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        cells,
    mesh::CellType type)
{
  /// Retrieve VTK permutation for given cell type
  std::vector<std::uint8_t> permutation
      = mesh::vtk_to_fenics(type, cells.cols());

  /// Permute input cells
  Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cells_dolfin(cells.rows(), cells.cols());
  for (int c = 0; c < cells_dolfin.rows(); ++c)
  {
    for (int v = 0; v < cells_dolfin.cols(); ++v)
    {
      cells_dolfin(c, v) = cells(c, permutation[v]);
    }
  }
  return cells_dolfin;
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> mesh::default_cell_permutation(mesh::CellType type,
                                                         std::int32_t degree)
{
  int n;
  switch (type)
  {
  case mesh::CellType::quadrilateral:
    // Quadrilateral cells follow lexciographic order (LG) and must be mapped
    // to tensor product ordering.
    n = (degree + 1) * (degree + 1);
    switch (degree)
    {
    case 1:
      // Current default for built in meshes
      return mesh::lexico_to_tp(type, n);
    default:
      // mesh::compute_local_to_global_point_map assumes that the first four
      // points in the connectivity array are the vertices, thus you need VTK
      // ordering.
      return mesh::vtk_cell_permutation(type, n);
    }

  case mesh::CellType::hexahedron:
    switch (degree)
    {
    case 1:
      // First order hexes follows lexiographic ordering
      return {0, 1, 2, 3, 4, 5, 6, 7};
    default:
      throw std::runtime_error("Higher order hexahedron not supported");
    }
    break;
  case mesh::CellType::interval:
    n = 2;
    break;
  case mesh::CellType::point:
    n = 1;
    break;
  case mesh::CellType::tetrahedron:
    n = 4;
    break;
  case mesh::CellType::triangle:
    n = (degree + 1) * (degree + 2) / 2;
    break;
  default:
    throw std::runtime_error("Unknown cell type.");
  }
  return mesh::vtk_cell_permutation(type, n);
}
//-----------------------------------------------------------------------------
