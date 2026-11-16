// Copyright (C) 2019-2026 Jorgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "cells.h"
#include <algorithm>
#include <array>
#include <basix/cell.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/cell_types.h>
#include <format>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace
{
//----------------------------------------------------------------------------
std::uint16_t vec_pop(std::vector<std::uint16_t>& v, int i)
{
  auto pos = (i < 0) ? v.end() + i : v.begin() + i;
  std::uint16_t value = *pos;
  v.erase(pos);
  return value;
}
//----------------------------------------------------------------------------
std::vector<std::uint16_t>
vtk_triangle_remainders(std::vector<std::uint16_t> remainders)
{
  std::vector<std::uint16_t> map;
  map.reserve(remainders.size());

  while (!remainders.empty())
  {
    if (remainders.size() == 1)
    {
      map.push_back(vec_pop(remainders, 0));
      break;
    }

    int degree
        = io::cells::cell_degree(mesh::CellType::triangle, remainders.size());

    map.push_back(vec_pop(remainders, 0));
    map.push_back(vec_pop(remainders, degree - 1));
    map.push_back(vec_pop(remainders, -1));

    for (int i = 0; i < degree - 1; ++i)
      map.push_back(vec_pop(remainders, 0));

    for (int i = 1, k = degree * (degree - 1) / 2; i < degree;
         k -= degree - i, ++i)
    {
      map.push_back(vec_pop(remainders, -k));
    }

    for (int i = 1, k = 1; i < degree; k += i, ++i)
      map.push_back(vec_pop(remainders, -k));
  }

  return map;
}
//-----------------------------------------------------------------------------
std::vector<std::uint16_t> vtk_triangle(int num_nodes)
{
  std::vector<std::uint16_t> map;
  map.reserve(num_nodes);

  // Vertices
  map.insert(map.begin(), {0, 1, 2});

  int degree = io::cells::cell_degree(mesh::CellType::triangle, num_nodes);
  for (int k = 1; k < degree; ++k)
    map.push_back(3 + 2 * (degree - 1) + k - 1);
  for (int k = 1; k < degree; ++k)
    map.push_back(3 + k - 1);
  for (int k = 1; k < degree; ++k)
    map.push_back(2 * degree - (k - 1));

  if (degree < 3)
    return map;

  // Interior VTK is ordered as a lower-order triangle, while DOLFINx
  // orders them lexicographically
  std::vector<std::uint16_t> rem(num_nodes - map.size());
  std::iota(rem.begin(), rem.end(), 3 * degree);
  std::ranges::copy(vtk_triangle_remainders(rem), std::back_inserter(map));

  return map;
}
//-----------------------------------------------------------------------------
std::vector<std::uint16_t>
vtk_tetrahedron_remainders(std::vector<std::uint16_t> remainders)
{
  std::vector<std::uint16_t> map;
  map.reserve(remainders.size());

  while (!remainders.empty())
  {
    if (remainders.size() == 1)
    {
      map.push_back(vec_pop(remainders, 0));
      break;
    }

    const int deg
        = io::cells::cell_degree(mesh::CellType::tetrahedron, remainders.size())
          + 1;
    map.push_back(vec_pop(remainders, 0));
    map.push_back(vec_pop(remainders, deg - 2));
    map.push_back(vec_pop(remainders, deg * (deg + 1) / 2 - 3));
    map.push_back(vec_pop(remainders, -1));

    if (deg > 2)
    {
      for (int i = 0; i < deg - 2; ++i)
        map.push_back(vec_pop(remainders, 0));

      {
        int d = deg - 2;
        for (int i = 0; i < deg - 2; ++i)
        {
          map.push_back(vec_pop(remainders, d));
          d += (deg - 3 - i);
        }
      }

      {
        int d = (deg - 2) * (deg - 1) / 2 - 1;
        for (int i = 0; i < deg - 2; ++i)
        {
          map.push_back(vec_pop(remainders, d));
          d -= (2 + i);
        }
      }

      {
        int d = (deg - 3) * (deg - 2) / 2;
        for (int i = 0; i < deg - 2; ++i)
        {
          map.push_back(vec_pop(remainders, d));
          d += ((deg - i) * (deg - i - 1) / 2 - 1);
        }
      }

      {
        int d = (deg - 3) * (deg - 2) / 2 + deg - 3;
        for (int i = 0; i < deg - 2; ++i)
        {
          map.push_back(vec_pop(remainders, d));
          d += ((deg - 2 - i) * (deg - 1 - i) / 2 + deg - 4 - i);
        }
      }

      {
        int d = (deg - 3) * (deg - 2) / 2 + deg - 3 + (deg - 2) * (deg - 1) / 2
                - 1;
        for (int i = 0; i < deg - 2; ++i)
        {
          map.push_back(vec_pop(remainders, d));
          d += ((deg - 3 - i) * (deg - 2 - i) / 2 + deg - i - 5);
        }
      }
    }

    if (deg > 3)
    {
      {
        std::vector<std::uint16_t> dofs;
        int d = (deg - 3) * (deg - 2) / 2;
        for (int i = 0; i < deg - 3; ++i)
        {
          for (int ii = 0; ii < deg - 3 - i; ++ii)
            dofs.push_back(vec_pop(remainders, d));
          d += ((deg - 2 - i) * (deg - 1 - i) / 2 - 1);
        }

        std::ranges::copy(vtk_triangle_remainders(dofs),
                          std::back_inserter(map));
      }

      {
        std::vector<std::uint16_t> dofs;
        int start = deg * deg - 4 * deg + 2;
        int sub_i_start = deg - 3;
        for (int i = 0; i < deg - 3; ++i)
        {
          int d = start;
          int sub_i = sub_i_start;
          for (int ii = 0; ii < deg - 3 - i; ++ii)
          {
            dofs.push_back(vec_pop(remainders, d));
            d += sub_i * (sub_i + 1) / 2 - 2 - i;
            sub_i -= 1;
          }

          start -= (2 + i);
        }

        std::ranges::copy(vtk_triangle_remainders(dofs),
                          std::back_inserter(map));
      }

      {
        std::vector<std::uint16_t> dofs;
        int start = (deg - 3) * (deg - 2) / 2;
        int sub_i_start = deg - 3;
        for (int i = 0; i < deg - 3; ++i)
        {
          int d = start;
          int sub_i = sub_i_start;
          for (int ii = 0; ii < deg - 3 - i; ++ii)
          {
            dofs.push_back(vec_pop(remainders, d));
            d += sub_i * (sub_i + 1) / 2 - 1 - 2 * i;
            sub_i -= 1;
          }

          start += (deg - 4 - i);
        }

        std::ranges::copy(vtk_triangle_remainders(dofs),
                          std::back_inserter(map));
      }

      {
        std::vector<std::uint16_t> dofs;
        int add_start = deg - 4;
        for (int i = 0; i < deg - 3; ++i)
        {
          int d = 0;
          int add = add_start;
          for (int ii = 0; ii < deg - 3 - i; ++ii)
          {
            dofs.push_back(vec_pop(remainders, d));
            d += add;
            add -= 1;
          }

          add_start -= 1;
        }

        std::ranges::copy(vtk_triangle_remainders(dofs),
                          std::back_inserter(map));
      }
    }
  }

  return map;
}
//-----------------------------------------------------------------------------
std::vector<std::uint16_t> vtk_tetrahedron(int num_nodes)
{
  const int degree
      = io::cells::cell_degree(mesh::CellType::tetrahedron, num_nodes);

  std::vector<std::uint16_t> map;
  map.reserve(num_nodes);

  // Vertices
  map.insert(map.begin(), {0, 1, 2, 3});

  if (degree < 2)
    return map;

  int base = 4;
  const int edge_dofs = degree - 1;
  for (int edge : {5, 2, 4, 3, 1, 0})
  {
    if (edge == 4)
    {
      for (int i = 0; i < edge_dofs; ++i)
        map.push_back(base + edge_dofs * (edge + 1) - 1 - i);
    }
    else
    {
      for (int i = 0; i < edge_dofs; ++i)
        map.push_back(base + edge_dofs * edge + i);
    }
  }

  if (degree < 3)
    return map;

  base += 6 * edge_dofs;
  const int n_face_dofs = (degree - 1) * (degree - 2) / 2;
  for (int face : {2, 0, 1, 3})
  {
    std::vector<std::uint16_t> face_dofs;
    face_dofs.reserve(n_face_dofs);
    if (face == 2)
    {
      for (int i = 0; i < n_face_dofs; ++i)
        face_dofs.push_back(base + n_face_dofs * face + i);
    }
    else if (face == 0)
    {
      for (int i = degree - 3; i >= 0; --i)
      {
        int d = i;
        for (int ii = 0; ii <= i; ++ii)
        {
          face_dofs.push_back(base + n_face_dofs * face + d);
          d += degree - 3 - ii;
        }
      }
    }
    else
    {
      for (int i = 0; i < degree - 2; ++i)
      {
        int d = i;
        for (int ii = 0; ii < degree - 2 - i; ++ii)
        {
          face_dofs.push_back(base + n_face_dofs * face + d);
          d += degree - 2 - ii;
        }
      }
    }

    std::ranges::copy(vtk_triangle_remainders(face_dofs),
                      std::back_inserter(map));
  }

  if (degree < 4)
    return map;

  base += 4 * n_face_dofs;
  std::vector<std::uint16_t> remainders((degree - 1) * (degree - 2)
                                        * (degree - 3) / 6);
  std::iota(remainders.begin(), remainders.end(), base);
  std::ranges::copy(vtk_tetrahedron_remainders(remainders),
                    std::back_inserter(map));

  return map;
}
//-----------------------------------------------------------------------------
std::vector<std::uint16_t> vtk_wedge(int num_nodes)
{
  switch (num_nodes)
  {
  case 6:
    return {0, 1, 2, 3, 4, 5};
  case 15:
    return {0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11};
  case 18:
    return {0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11, 15, 17, 16};
  default:
    throw std::invalid_argument(std::format(
        "VTK wedges with {} nodes are not supported, expected 6, 15 or 18.",
        num_nodes));
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint16_t> vtk_pyramid(int num_nodes)
{
  switch (num_nodes)
  {
  case 5:
    return {0, 1, 3, 2, 4};
  case 13:
    return {0, 1, 3, 2, 4, 5, 8, 10, 6, 7, 9, 12, 11};
  case 14:
    return {0, 1, 3, 2, 4, 5, 8, 10, 6, 7, 9, 12, 11, 13};
  default:
    throw std::invalid_argument(std::format(
        "VTK pyramids with {} nodes are not supported, expected 5, 13 or 14.",
        num_nodes));
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint16_t> vtk_quadrilateral(int num_nodes)
{
  // Degree-2 serendipity: a node on each vertex and each edge and none
  // inside, which is not a Lagrange layout, so cell_degree below has no
  // answer for it. VTK takes the edges in the order DOLFINx numbers
  // 0, 2, 3, 1.
  if (num_nodes == 8)
    return {0, 1, 3, 2, 4, 6, 7, 5};

  std::vector<std::uint16_t> map;
  map.reserve(num_nodes);

  // Vertices
  map.insert(map.begin(), {0, 1, 3, 2});

  const int n
      = io::cells::cell_degree(mesh::CellType::quadrilateral, num_nodes);

  // Edges
  const int edge_nodes = n - 1;
  for (int k = 0; k < edge_nodes; ++k)
    map.push_back(4 + k);
  for (int k = 0; k < edge_nodes; ++k)
    map.push_back(4 + 2 * edge_nodes + k);
  for (int k = 0; k < edge_nodes; ++k)
    map.push_back(4 + 3 * edge_nodes + k);
  for (int k = 0; k < edge_nodes; ++k)
    map.push_back(4 + edge_nodes + k);

  // Face
  for (int k = 0; k < edge_nodes * edge_nodes; ++k)
    map.push_back(4 + edge_nodes * 4 + k);

  return map;
}
//-----------------------------------------------------------------------------
std::vector<std::uint16_t> vtk_hexahedron(int num_nodes)
{
  if (num_nodes == 20)
  {
    std::vector<std::uint16_t> map;
    map.insert(map.begin(), {0, 1, 3, 2, 4, 5, 7, 6});
    for (int e : {0, 3, 5, 1, 8, 10, 11, 9, 2, 4, 7, 6})
      map.push_back(8 + e);
    return map;
  }

  std::uint16_t n
      = io::cells::cell_degree(mesh::CellType::hexahedron, num_nodes);

  std::vector<std::uint16_t> map;
  map.reserve(num_nodes);

  // Vertices
  map.insert(map.begin(), {0, 1, 3, 2, 4, 5, 7, 6});

  // Edges
  int base = 8;
  const int edge_nodes = n - 1;
  for (int e : {0, 3, 5, 1, 8, 10, 11, 9, 2, 4, 7, 6})
    for (int i = 0; i < edge_nodes; ++i)
      map.push_back(base + edge_nodes * e + i);

  base += 12 * edge_nodes;
  const int face_nodes = edge_nodes * edge_nodes;
  for (int f : {2, 3, 1, 4, 0, 5})
    for (int i = 0; i < face_nodes; ++i)
      map.push_back(base + face_nodes * f + i);

  // Volume
  base += 6 * face_nodes;
  for (int i = 0; i < face_nodes * edge_nodes; ++i)
    map.push_back(base + i);

  return map;
}
//-----------------------------------------------------------------------------
/// @brief Position of a node on the lattice of a cell of degree `k`.
///
/// The coordinates of the node in the affine frame of the Basix
/// reference cell (see frame_coordinates), times `k`. On a cell that is
/// an affine image of a unit cell these are integers. Coordinates beyond
/// the topological dimension are zero.
using LatticePoint = std::array<int, 3>;
//-----------------------------------------------------------------------------
LatticePoint operator+(const LatticePoint& a, const LatticePoint& b)
{
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
//-----------------------------------------------------------------------------
LatticePoint operator-(const LatticePoint& a, const LatticePoint& b)
{
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
//-----------------------------------------------------------------------------
LatticePoint operator*(int s, const LatticePoint& a)
{
  return {s * a[0], s * a[1], s * a[2]};
}
//-----------------------------------------------------------------------------
/// @brief Coordinates of points in the affine frame of a Basix reference
/// cell.
///
/// The frame has its origin at vertex 0 and one axis along each edge at
/// vertex 0, in the Basix edge order. It is taken from the Basix cell
/// geometry and topology, so it follows any change of the reference
/// cell.
///
/// Example: the Basix quadrilateral has the vertices x_0 = (0, 0),
/// x_1 = (1, 0), x_2 = (0, 1) and x_3 = (1, 1). The axes are
/// a_0 = x_1 - x_0 = (1, 0) and a_1 = x_2 - x_0 = (0, 1), and a point x
/// has the frame coordinates c with x = x_0 + c_0 a_0 + c_1 a_1. The cell
/// midpoint (1/2, 1/2) has frame coordinates (1/2, 1/2), and x_3 has
/// (1, 1). Were the reference cell [-1, 1]^2 instead, with
/// x_0 = (-1, -1), x_1 = (1, -1) and x_2 = (-1, 1), the axes would be
/// (2, 0) and (0, 2), and the midpoint (0, 0) and x_3 = (1, 1) would have
/// the same frame coordinates (1/2, 1/2) and (1, 1).
///
/// @param[in] cell Cell type.
/// @param[in] x Points on the reference cell, row-major, shape
/// `(num_points, tdim)`.
/// @return Frame coordinates of the points, row-major, shape
/// `(num_points, tdim)`.
std::vector<double> frame_coordinates(mesh::CellType cell,
                                      std::span<const double> x)
{
  const basix::cell::type basix_cell = mesh::cell_type_to_basix_type(cell);
  const auto [g, gshape] = basix::cell::geometry<double>(basix_cell);
  const std::size_t tdim = gshape[1];

  // Vertices at the other end of the edges at vertex 0
  const std::vector<std::vector<std::vector<int>>> topology
      = basix::cell::topology(basix_cell);
  std::vector<int> axes;
  for (const std::vector<int>& e : topology[1])
  {
    if (e[0] == 0)
      axes.push_back(e[1]);
    else if (e[1] == 0)
      axes.push_back(e[0]);
  }
  assert(axes.size() == tdim);

  // Solve A c = x - x_0 for all points, with the axes as the columns of
  // A, by Gauss-Jordan elimination with partial pivoting (tdim <= 3)
  std::vector<double> A(tdim * tdim);
  for (std::size_t i = 0; i < tdim; ++i)
    for (std::size_t j = 0; j < tdim; ++j)
      A[i * tdim + j] = g[axes[j] * tdim + i] - g[i];
  const std::size_t num_points = x.size() / tdim;
  std::vector<double> b(tdim * num_points);
  for (std::size_t p = 0; p < num_points; ++p)
    for (std::size_t i = 0; i < tdim; ++i)
      b[i * num_points + p] = x[p * tdim + i] - g[i];
  for (std::size_t col = 0; col < tdim; ++col)
  {
    std::size_t pivot = col;
    for (std::size_t r = col + 1; r < tdim; ++r)
      if (std::abs(A[r * tdim + col]) > std::abs(A[pivot * tdim + col]))
        pivot = r;
    for (std::size_t j = 0; j < tdim; ++j)
      std::swap(A[col * tdim + j], A[pivot * tdim + j]);
    for (std::size_t p = 0; p < num_points; ++p)
      std::swap(b[col * num_points + p], b[pivot * num_points + p]);
    for (std::size_t r = 0; r < tdim; ++r)
    {
      if (r == col)
        continue;
      const double factor = A[r * tdim + col] / A[col * tdim + col];
      for (std::size_t j = 0; j < tdim; ++j)
        A[r * tdim + j] -= factor * A[col * tdim + j];
      for (std::size_t p = 0; p < num_points; ++p)
        b[r * num_points + p] -= factor * b[col * num_points + p];
    }
  }

  std::vector<double> coordinates(num_points * tdim);
  for (std::size_t p = 0; p < num_points; ++p)
    for (std::size_t i = 0; i < tdim; ++i)
      coordinates[p * tdim + i] = b[i * num_points + p] / A[i * tdim + i];
  return coordinates;
}
//-----------------------------------------------------------------------------
/// @brief Lattice points of points on the Basix reference cell.
/// @param[in] cell Cell type.
/// @param[in] x Points on the reference cell, row-major, shape
/// `(num_points, tdim)`.
/// @param[in] k Degree of the lattice.
/// @return The lattice point of each point.
std::vector<LatticePoint> lattice_points(mesh::CellType cell,
                                         std::span<const double> x, int k)
{
  const std::size_t tdim = mesh::cell_dim(cell);
  const std::vector<double> c = frame_coordinates(cell, x);
  std::vector<LatticePoint> points(c.size() / tdim, {0, 0, 0});
  for (std::size_t p = 0; p < points.size(); ++p)
  {
    for (std::size_t i = 0; i < tdim; ++i)
    {
      const double ci = k * c[p * tdim + i];
      assert(std::abs(ci - std::round(ci)) < 1e-8);
      points[p][i] = static_cast<int>(std::lround(ci));
    }
  }
  return points;
}
//-----------------------------------------------------------------------------
/// @brief Gmsh's vertex numbering of a cell.
///
/// Gmsh's own reference cells (e.g. [-1, 1]^2) are not used: the node
/// order depends only on which vertex Gmsh gives each number, which an
/// affine map between reference cells keeps. For example, Gmsh numbers
/// the quadrilateral counter-clockwise, so its vertex 2 is the Basix
/// vertex 3.
///
/// @param[in] cell Cell type.
/// @return The Basix vertex of each Gmsh vertex, in Gmsh order.
std::vector<int> gmsh_vertices(mesh::CellType cell)
{
  switch (cell)
  {
  case mesh::CellType::interval:
    return {0, 1};
  case mesh::CellType::triangle:
    return {0, 1, 2};
  case mesh::CellType::quadrilateral:
    return {0, 1, 3, 2};
  case mesh::CellType::tetrahedron:
    return {0, 1, 2, 3};
  case mesh::CellType::hexahedron:
    return {0, 1, 3, 2, 4, 5, 7, 6};
  case mesh::CellType::prism:
    return {0, 1, 2, 3, 4, 5};
  case mesh::CellType::pyramid:
    return {0, 1, 3, 2, 4};
  default:
    throw std::invalid_argument("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
/// @brief Lattice points of Gmsh's vertices at degree 1.
/// @param[in] cell Cell type.
/// @return The lattice point of each Gmsh vertex, in Gmsh order.
std::vector<LatticePoint> gmsh_vertex_points(mesh::CellType cell)
{
  const basix::cell::type basix_cell = mesh::cell_type_to_basix_type(cell);
  const auto [g, gshape] = basix::cell::geometry<double>(basix_cell);
  const std::vector<LatticePoint> basix_vertices = lattice_points(cell, g, 1);
  std::vector<LatticePoint> v;
  for (int i : gmsh_vertices(cell))
    v.push_back(basix_vertices[i]);
  return v;
}
//-----------------------------------------------------------------------------
/// @brief Gmsh's edges of a cell, in Gmsh order.
/// @param[in] cell Cell type.
/// @return The edges as pairs of Gmsh vertices. The nodes inside an edge
/// run from its first vertex to its second.
std::vector<std::array<int, 2>> gmsh_edges(mesh::CellType cell)
{
  switch (cell)
  {
  case mesh::CellType::interval:
    return {{0, 1}};
  case mesh::CellType::triangle:
    return {{0, 1}, {1, 2}, {2, 0}};
  case mesh::CellType::quadrilateral:
    return {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
  case mesh::CellType::tetrahedron:
    return {{0, 1}, {1, 2}, {2, 0}, {3, 0}, {3, 2}, {3, 1}};
  case mesh::CellType::hexahedron:
    return {{0, 1}, {0, 3}, {0, 4}, {1, 2}, {1, 5}, {2, 3},
            {2, 6}, {3, 7}, {4, 5}, {4, 7}, {5, 6}, {6, 7}};
  case mesh::CellType::prism:
    return {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 4},
            {2, 5}, {3, 4}, {3, 5}, {4, 5}};
  case mesh::CellType::pyramid:
    return {{0, 1}, {0, 3}, {0, 4}, {1, 2}, {1, 4}, {2, 3}, {2, 4}, {3, 4}};
  default:
    throw std::invalid_argument("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
/// @brief Gmsh's faces of a 3D cell, in Gmsh order.
/// @param[in] cell Cell type.
/// @return The faces as lists of Gmsh vertices, empty for cells of
/// dimension below 3. The nodes inside a face are ordered as a Gmsh
/// triangle or quadrilateral whose first vertex is the face's first
/// vertex, and whose first two edges point to the face's second and last
/// vertices.
std::vector<std::vector<int>> gmsh_faces(mesh::CellType cell)
{
  switch (cell)
  {
  case mesh::CellType::tetrahedron:
    return {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {3, 1, 2}};
  case mesh::CellType::hexahedron:
    return {{0, 3, 2, 1}, {0, 1, 5, 4}, {0, 4, 7, 3},
            {1, 2, 6, 5}, {2, 3, 7, 6}, {4, 5, 6, 7}};
  case mesh::CellType::prism:
    return {{0, 2, 1}, {3, 4, 5}, {0, 1, 4, 3}, {0, 3, 5, 2}, {1, 2, 5, 4}};
  case mesh::CellType::pyramid:
    return {{0, 1, 4}, {3, 0, 4}, {1, 2, 4}, {2, 3, 4}, {0, 3, 2, 1}};
  default:
    return {};
  }
}
//-----------------------------------------------------------------------------
/// @brief Coordinates of a lattice point of a Gmsh triangle or
/// quadrilateral along its edges from vertex 0 to vertex 1 and to its last
/// vertex.
/// @param[in] cell Triangle or quadrilateral.
/// @param[in] p Lattice point of a cell of degree `m`.
/// @param[in] m Degree of the cell.
/// @return The coordinates `(a, b)` such that `p` is `m` times vertex 0
/// plus `a` times the first edge plus `b` times the second.
std::array<int, 2> gmsh_edge_coordinates(mesh::CellType cell,
                                         const LatticePoint& p, int m)
{
  const std::vector<LatticePoint> v = gmsh_vertex_points(cell);
  const LatticePoint e0 = v[1] - v[0];
  const LatticePoint e1 = v.back() - v[0];
  const LatticePoint q = p - m * v[0];
  const int det = e0[0] * e1[1] - e0[1] * e1[0];
  return {(q[0] * e1[1] - q[1] * e1[0]) / det,
          (e0[0] * q[1] - e0[1] * q[0]) / det};
}
//-----------------------------------------------------------------------------
/// @brief Gmsh's node order of a Lagrange cell.
///
/// Gmsh numbers the vertices, then the nodes inside each edge (see
/// gmsh_edges) and each face (see gmsh_faces), then the interior nodes.
/// The interior nodes are ordered as a cell of the same type and lower
/// degree, moved one step along each edge at vertex 0. Prisms are the
/// exception: their interior is the interior of the base triangle times
/// the interior of the vertical edge from vertex 0.
///
/// @param[in] cell Cell type.
/// @param[in] k Degree of the cell.
/// @return The lattice points of the nodes, in Gmsh order.
std::vector<LatticePoint> gmsh_lattice_points(mesh::CellType cell, int k)
{
  const std::vector<LatticePoint> v = gmsh_vertex_points(cell);
  if (k == 0)
    return {v[0]};

  std::vector<LatticePoint> points;
  for (const LatticePoint& x : v)
    points.push_back(k * x);

  for (auto [a, b] : gmsh_edges(cell))
    for (int t = 1; t < k; ++t)
      points.push_back(k * v[a] + t * (v[b] - v[a]));

  for (const std::vector<int>& f : gmsh_faces(cell))
  {
    const mesh::CellType face_type = f.size() == 4
                                         ? mesh::CellType::quadrilateral
                                         : mesh::CellType::triangle;
    const int m = k - (f.size() == 4 ? 2 : 3);
    if (m < 0)
      continue;
    const LatticePoint e0 = v[f[1]] - v[f[0]];
    const LatticePoint e1 = v[f.back()] - v[f[0]];
    for (const LatticePoint& p : gmsh_lattice_points(face_type, m))
    {
      auto [a, b] = gmsh_edge_coordinates(face_type, p, m);
      points.push_back(k * v[f[0]] + (a + 1) * e0 + (b + 1) * e1);
    }
  }

  switch (cell)
  {
  case mesh::CellType::interval:
    break; // The edge nodes are the interior nodes
  case mesh::CellType::prism:
  {
    if (k < 3)
      break;
    // The vertical edge is ordered as a line of degree k - 2: ends first
    std::vector<int> z = {0, k - 2};
    for (int t = 1; t < k - 2; ++t)
      z.push_back(t);
    for (const LatticePoint& p :
         gmsh_lattice_points(mesh::CellType::triangle, k - 3))
    {
      auto [a, b] = gmsh_edge_coordinates(mesh::CellType::triangle, p, k - 3);
      for (int zi : z)
      {
        points.push_back(k * v[0] + (a + 1) * (v[1] - v[0])
                         + (b + 1) * (v[2] - v[0]) + (zi + 1) * (v[3] - v[0]));
      }
    }
    break;
  }
  default:
  {
    int reduction = 2; // Quadrilateral, hexahedron
    if (cell == mesh::CellType::triangle or cell == mesh::CellType::pyramid)
      reduction = 3;
    else if (cell == mesh::CellType::tetrahedron)
      reduction = 4;
    if (k < reduction)
      break;

    // Move the lower-degree cell one step along each edge at vertex 0
    LatticePoint shift = reduction * v[0];
    for (auto [a, b] : gmsh_edges(cell))
    {
      if (a == 0)
        shift = shift + (v[b] - v[0]);
      else if (b == 0)
        shift = shift + (v[a] - v[0]);
    }
    for (const LatticePoint& p : gmsh_lattice_points(cell, k - reduction))
      points.push_back(p + shift);
  }
  }

  return points;
}
//-----------------------------------------------------------------------------
/// @brief DOLFINx's node order of a Lagrange cell.
///
/// Read from the equispaced Basix Lagrange element, so that it follows
/// any change of the Basix node layout.
///
/// @param[in] cell Cell type.
/// @param[in] k Degree of the cell.
/// @return The lattice points of the nodes, in DOLFINx order.
std::vector<LatticePoint> basix_lattice_points(mesh::CellType cell, int k)
{
  basix::FiniteElement<double> e = basix::create_element<double>(
      basix::element::family::P, mesh::cell_type_to_basix_type(cell), k,
      basix::element::lagrange_variant::equispaced,
      basix::element::dpc_variant::unset, false);
  return lattice_points(cell, e.points().first, k);
}
//-----------------------------------------------------------------------------
/// @brief Map from Gmsh to DOLFINx node numbers of a Lagrange cell.
///
/// Nodes are matched by their lattice points.
///
/// @param[in] cell Cell type.
/// @param[in] num_nodes Number of nodes of the cell.
/// @return The DOLFINx node of each Gmsh node.
/// @throws std::invalid_argument If no Lagrange cell of type `cell` has
/// `num_nodes` nodes.
std::vector<std::uint16_t> gmsh_lagrange(mesh::CellType cell, int num_nodes)
{
  const int k = io::cells::cell_degree(cell, num_nodes);
  const std::vector<LatticePoint> dolfinx = basix_lattice_points(cell, k);
  const std::vector<LatticePoint> gmsh = gmsh_lattice_points(cell, k);
  assert(gmsh.size() == dolfinx.size());

  std::vector<std::uint16_t> order(dolfinx.size());
  std::iota(order.begin(), order.end(), 0);
  auto point = [&dolfinx](std::uint16_t i) -> const LatticePoint&
  { return dolfinx[i]; };
  std::ranges::sort(order, std::less{}, point);
  std::vector<std::uint16_t> map;
  map.reserve(gmsh.size());
  for (const LatticePoint& x : gmsh)
  {
    auto it = std::ranges::lower_bound(order, x, std::less{}, point);
    assert(it != order.end() and dolfinx[*it] == x);
    map.push_back(*it);
  }

  return map;
}
} // namespace
//-----------------------------------------------------------------------------
std::vector<std::uint16_t> io::cells::perm_vtk(mesh::CellType type,
                                               int num_nodes)
{
  std::vector<std::uint16_t> map;
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
    throw std::invalid_argument("Unknown cell type.");
  }

  return cells::transpose(map);
}
//-----------------------------------------------------------------------------
std::vector<std::uint16_t> io::cells::perm_gmsh(mesh::CellType type,
                                                int num_nodes)
{
  if (num_nodes > std::numeric_limits<std::uint16_t>::max())
  {
    throw std::invalid_argument(
        std::format("Gmsh cells with {} nodes are not supported, at most {}.",
                    num_nodes, std::numeric_limits<std::uint16_t>::max()));
  }

  std::vector<std::uint16_t> map;
  if (type == mesh::CellType::point)
    map = {0};
  else if (type == mesh::CellType::prism and num_nodes == 15)
    map = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}; // Serendipity
  else if (type == mesh::CellType::pyramid and num_nodes == 13)
    map = {0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 12, 11}; // Serendipity
  else
    map = gmsh_lagrange(type, num_nodes);

  return cells::transpose(map);
}
//-----------------------------------------------------------------------------
int io::cells::cell_degree(mesh::CellType type, int num_nodes)
{
  switch (type)
  {
  case mesh::CellType::point:
    return 1;
  case mesh::CellType::interval:
    return num_nodes - 1;
  case mesh::CellType::triangle:
  {
    const int n = (std::sqrt(1 + 8 * num_nodes) - 1) / 2;
    if (2 * num_nodes != n * (n + 1))
    {
      throw std::invalid_argument(std::format(
          "Unknown triangle layout. Number of nodes: {}", num_nodes));
    }
    return n - 1;
  }
  case mesh::CellType::tetrahedron:
  {
    int n = 0;
    while (n * (n + 1) * (n + 2) < 6 * num_nodes)
      ++n;
    if (n * (n + 1) * (n + 2) != 6 * num_nodes)
    {
      throw std::invalid_argument(std::format(
          "Unknown tetrahedron layout. Number of nodes: {}", num_nodes));
    }
    return n - 1;
  }
  case mesh::CellType::quadrilateral:
  {
    const int n = std::sqrt(num_nodes);
    if (num_nodes != n * n)
    {
      throw std::invalid_argument(std::format(
          "Unknown quadrilateral layout. Number of nodes: {}", num_nodes));
    }
    return n - 1;
  }
  case mesh::CellType::hexahedron:
  {
    const int n = std::cbrt(num_nodes);
    if (num_nodes != n * n * n)
    {
      throw std::invalid_argument(std::format(
          "Unknown hexahedron layout. Number of nodes: {}", num_nodes));
    }
    return n - 1;
  }
  case mesh::CellType::prism:
  {
    // A degree n prism has (n + 1)^2 (n + 2) / 2 nodes
    int n = 1;
    while ((n + 1) * (n + 1) * (n + 2) < 2 * num_nodes)
      ++n;
    if ((n + 1) * (n + 1) * (n + 2) != 2 * num_nodes)
    {
      throw std::invalid_argument(std::format(
          "No Lagrange prism has {} nodes. The 15-node wedge is a "
          "serendipity layout, which Basix does not have for a prism.",
          num_nodes));
    }
    return n;
  }
  case mesh::CellType::pyramid:
  {
    // A degree n pyramid has (n + 1) (n + 2) (2n + 3) / 6 nodes
    int n = 1;
    while ((n + 1) * (n + 2) * (2 * n + 3) < 6 * num_nodes)
      ++n;
    if ((n + 1) * (n + 2) * (2 * n + 3) != 6 * num_nodes)
    {
      throw std::invalid_argument(std::format(
          "No Lagrange pyramid has {} nodes. The 13-node pyramid is a "
          "serendipity layout, which Basix does not have for a pyramid.",
          num_nodes));
    }
    return n;
  }
  default:
    throw std::invalid_argument("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint16_t>
io::cells::transpose(std::span<const std::uint16_t> map)
{
  std::vector<std::uint16_t> transpose(map.size());
  for (std::size_t i = 0; i < map.size(); ++i)
    transpose[map[i]] = i;
  return transpose;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
io::cells::apply_permutation(std::span<const std::int64_t> cells,
                             std::array<std::size_t, 2> shape,
                             std::span<const std::uint16_t> p)
{
  if (cells.size() != shape[0] * shape[1])
    throw std::invalid_argument("Cell array size does not match given shape.");
  assert(shape[1] == p.size());

  logger().info("IO permuting cells");
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
std::int8_t io::cells::get_vtk_cell_type(mesh::CellType cell, int num_nodes)
{
  // Arbitrary-degree Lagrange types where VTK has them
  // https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
  switch (cell)
  {
  case mesh::CellType::point:
    return 1;
  case mesh::CellType::interval:
    return 68;
  case mesh::CellType::triangle:
    return 69;
  case mesh::CellType::quadrilateral:
    if (num_nodes == 8)
      return 23; // VTK_QUADRATIC_QUAD, the quadratic serendipity layout
    if (num_nodes < 4)
      throw std::invalid_argument("VTK quadrilaterals need at least 4 nodes.");
    cell_degree(cell, num_nodes); // Throws if not a Lagrange layout
    return 70;
  case mesh::CellType::tetrahedron:
    return 71;
  case mesh::CellType::hexahedron:
    if (num_nodes == 20)
      return 25; // VTK_QUADRATIC_HEXAHEDRON, the quadratic serendipity layout
    if (num_nodes < 8)
      throw std::invalid_argument("VTK hexahedra need at least 8 nodes.");
    cell_degree(cell, num_nodes); // Throws if not a Lagrange layout
    return 72;
  case mesh::CellType::prism:
    return 73;
  case mesh::CellType::pyramid:
    // VTK has no arbitrary-degree Lagrange pyramid to fall back on, and
    // its quadratic pyramid (27) is the 13-node serendipity cell, which
    // basix has no element for, so it cannot be written or read back
    if (num_nodes == 5)
      return 14;
    throw std::invalid_argument(
        std::format("VTK pyramid output supports 5 nodes, got {}. "
                    "Lagrange pyramids are not implemented in VTK.",
                    num_nodes));
  default:
    throw std::invalid_argument("Unknown cell type");
  }
}
//----------------------------------------------------------------------------
