// Copyright (C) 2019 Matthew Scroggs
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dofs_permutation.h"
#include "ElementDofLayout.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Topology.h>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
int get_num_permutations(mesh::CellType cell_type)
{
  // In general, this will return num_edges + 2*num_faces + 4*num_volumes
  switch (cell_type)
  {
  case (mesh::CellType::point):
    return 0;
  case (mesh::CellType::interval):
    return 0;
  case (mesh::CellType::triangle):
    return 3;
  case (mesh::CellType::tetrahedron):
    return 14;
  case (mesh::CellType::quadrilateral):
    return 4;
  case (mesh::CellType::hexahedron):
    return 24;
  default:
    LOG(WARNING) << "Dof permutations are not defined for this cell type. High "
                    "order elements may be incorrect.";
    return 0;
  }
}
//-----------------------------------------------------------------------------
/// Calculates the number of times the rotation and reflection of a triangle
/// should be applied to a triangle with the given global vertex numbers
/// @param[in] v1, v2, v3 The global vertex numbers of the triangle's vertices
/// @return The rotation and reflection orders for the triangle
template <typename T>
std::array<std::int8_t, 2> calculate_triangle_orders(T v1, T v2, T v3)
{
  if (v1 < v2 and v1 < v3)
    return {0, v2 > v3};
  else if (v2 < v1 and v2 < v3)
    return {1, v3 > v1};
  else if (v3 < v1 and v3 < v2)
    return {2, v1 > v2};

  throw std::runtime_error("Two of a triangle's vertices appear to be equal.");
}
//-----------------------------------------------------------------------------
/// Calculates the number of times the rotation and reflection of a
/// quadrilateral should be applied to a quadrilateral with the given global
/// vertex numbers
/// @param[in] v1, v2, v3, v4 The global vertex numbers of the quadrilateral's
/// vertices
/// @return The rotation and reflection orders for the quadrilateral
template <typename T>
std::array<std::int8_t, 2> calculate_quadrilateral_orders(T v1, T v2, T v3,
                                                          T v4)
{
  if (v1 < v2 and v1 < v3 and v1 < v4)
    return {0, v2 > v3};
  else if (v2 < v1 and v2 < v3 and v2 < v4)
    return {1, v4 > v1};
  else if (v4 < v1 and v4 < v2 and v4 < v3)
    return {2, v3 > v2};
  else if (v3 < v1 and v3 < v2 and v3 < v4)
    return {3, v1 > v4};

  throw std::runtime_error(
      "Two of a quadrilateral's vertices appear to be equal.");
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
empty_ordering(const mesh::Topology& topology)
{
  const int D = topology.dim();
  auto cells = topology.connectivity(D, 0);
  assert(cells);
  const int num_cells = cells->num_nodes();
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_orders(num_cells, 0);
  return cell_orders;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
compute_ordering_triangle(const mesh::Topology& topology)
{
  const int D = topology.dim();
  auto cells = topology.connectivity(D, 0);
  assert(cells);
  const int num_cells = cells->num_nodes();
  const int num_permutations = get_num_permutations(topology.cell_type());
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_orders(num_cells, num_permutations);

  auto map = topology.index_map(0);
  assert(map);
  const std::vector global_indices = map->global_indices();

  // Set orders for each cell
  for (int cell_n = 0; cell_n < num_cells; ++cell_n)
  {
    auto vertices = cells->links(cell_n);
    const std::int64_t v0 = global_indices[vertices[0]];
    const std::int64_t v1 = global_indices[vertices[1]];
    const std::int64_t v2 = global_indices[vertices[2]];

    // Set the orders for the edge flips
    cell_orders(cell_n, 0) = (v1 > v2);
    cell_orders(cell_n, 1) = (v0 > v2);
    cell_orders(cell_n, 2) = (v0 > v1);
  }

  return cell_orders;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
compute_ordering_quadrilateral(const mesh::Topology& topology)
{
  const int D = topology.dim();
  auto cells = topology.connectivity(D, 0);
  assert(cells);
  const int num_cells = cells->num_nodes();
  const int num_permutations = get_num_permutations(topology.cell_type());
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_orders(num_cells, num_permutations);

  // Set orders for each cell
  auto map = topology.index_map(0);
  assert(map);
  const std::vector global_indices = map->global_indices();
  for (int cell_n = 0; cell_n < num_cells; ++cell_n)
  {
    auto vertices = cells->links(cell_n);
    const std::int64_t v0 = global_indices[vertices[0]];
    const std::int64_t v1 = global_indices[vertices[1]];
    const std::int64_t v2 = global_indices[vertices[2]];
    const std::int64_t v3 = global_indices[vertices[3]];

    // Set the orders for the edge flips
    cell_orders(cell_n, 0) = (v0 > v1);
    cell_orders(cell_n, 1) = (v2 > v3);
    cell_orders(cell_n, 2) = (v0 > v2);
    cell_orders(cell_n, 3) = (v1 > v3);
  }

  return cell_orders;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
compute_ordering_tetrahedron(const mesh::Topology& topology)
{
  const int D = topology.dim();
  auto cells = topology.connectivity(D, 0);
  assert(cells);
  const int num_cells = cells->num_nodes();
  const int num_permutations = get_num_permutations(topology.cell_type());
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_orders(num_cells, num_permutations);

  // Set orders for each cell
  auto map = topology.index_map(0);
  assert(map);
  const std::vector global_indices = map->global_indices();
  for (int cell_n = 0; cell_n < num_cells; ++cell_n)
  {
    auto vertices = cells->links(cell_n);
    const std::int64_t v0 = global_indices[vertices[0]];
    const std::int64_t v1 = global_indices[vertices[1]];
    const std::int64_t v2 = global_indices[vertices[2]];
    const std::int64_t v3 = global_indices[vertices[3]];

    // Set the orders for the edge flips
    cell_orders(cell_n, 0) = (v2 > v3);
    cell_orders(cell_n, 1) = (v1 > v3);
    cell_orders(cell_n, 2) = (v1 > v2);
    cell_orders(cell_n, 3) = (v0 > v3);
    cell_orders(cell_n, 4) = (v0 > v2);
    cell_orders(cell_n, 5) = (v0 > v1);

    // Set the orders for the face rotations and reflections
    const std::array tri_orders0 = calculate_triangle_orders(v1, v2, v3);
    cell_orders(cell_n, 6) = tri_orders0[0];
    cell_orders(cell_n, 7) = tri_orders0[1];
    const std::array tri_orders1 = calculate_triangle_orders(v0, v2, v3);
    cell_orders(cell_n, 8) = tri_orders1[0];
    cell_orders(cell_n, 9) = tri_orders1[1];
    const std::array tri_orders2 = calculate_triangle_orders(v0, v1, v3);
    cell_orders(cell_n, 10) = tri_orders2[0];
    cell_orders(cell_n, 11) = tri_orders2[1];
    const std::array tri_orders3 = calculate_triangle_orders(v0, v1, v2);
    cell_orders(cell_n, 12) = tri_orders3[0];
    cell_orders(cell_n, 13) = tri_orders3[1];
  }

  return cell_orders;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
compute_ordering_hexahedron(const mesh::Topology& topology)
{
  const int D = topology.dim();
  auto cells = topology.connectivity(D, 0);
  assert(cells);
  const int num_cells = cells->num_nodes();
  const int num_permutations = get_num_permutations(topology.cell_type());
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_orders(num_cells, num_permutations);

  // Set orders for each cell
  auto map = topology.index_map(0);
  assert(map);
  const std::vector global_indices = map->global_indices();
  for (int cell_n = 0; cell_n < num_cells; ++cell_n)
  {
    auto vertices = cells->links(cell_n);
    const std::int64_t v0 = global_indices[vertices[0]];
    const std::int64_t v1 = global_indices[vertices[1]];
    const std::int64_t v2 = global_indices[vertices[2]];
    const std::int64_t v3 = global_indices[vertices[3]];
    const std::int64_t v4 = global_indices[vertices[4]];
    const std::int64_t v5 = global_indices[vertices[5]];
    const std::int64_t v6 = global_indices[vertices[6]];
    const std::int64_t v7 = global_indices[vertices[7]];

    // Set the orders for the edge flips
    cell_orders(cell_n, 0) = (v0 > v1);
    cell_orders(cell_n, 1) = (v2 > v3);
    cell_orders(cell_n, 2) = (v4 > v5);
    cell_orders(cell_n, 3) = (v6 > v7);
    cell_orders(cell_n, 4) = (v0 > v2);
    cell_orders(cell_n, 5) = (v1 > v3);
    cell_orders(cell_n, 6) = (v4 > v6);
    cell_orders(cell_n, 7) = (v5 > v7);
    cell_orders(cell_n, 8) = (v0 > v4);
    cell_orders(cell_n, 9) = (v1 > v5);
    cell_orders(cell_n, 10) = (v2 > v6);
    cell_orders(cell_n, 11) = (v3 > v7);

    // Set the orders for the face rotations and reflections
    const std::array quad_orders0
        = calculate_quadrilateral_orders(v0, v1, v2, v3);
    cell_orders(cell_n, 12) = quad_orders0[0];
    cell_orders(cell_n, 13) = quad_orders0[1];
    const std::array quad_orders1
        = calculate_quadrilateral_orders(v4, v5, v6, v7);
    cell_orders(cell_n, 14) = quad_orders1[0];
    cell_orders(cell_n, 15) = quad_orders1[1];
    const std::array quad_orders2
        = calculate_quadrilateral_orders(v0, v1, v4, v5);
    cell_orders(cell_n, 16) = quad_orders2[0];
    cell_orders(cell_n, 17) = quad_orders2[1];
    const std::array quad_orders3
        = calculate_quadrilateral_orders(v2, v3, v6, v7);
    cell_orders(cell_n, 18) = quad_orders3[0];
    cell_orders(cell_n, 19) = quad_orders3[1];
    const std::array quad_orders4
        = calculate_quadrilateral_orders(v0, v2, v4, v6);
    cell_orders(cell_n, 20) = quad_orders4[0];
    cell_orders(cell_n, 21) = quad_orders4[1];
    const std::array quad_orders5
        = calculate_quadrilateral_orders(v1, v3, v5, v7);
    cell_orders(cell_n, 22) = quad_orders5[0];
    cell_orders(cell_n, 23) = quad_orders5[1];
  }

  return cell_orders;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
fem::compute_dof_permutations(const mesh::Topology& topology,
                              const fem::ElementDofLayout& dof_layout)
{
  // Build ordering in each cell. It stores the number of times each row
  // of _permutations should be applied on each cell Will have shape
  // (number of cells) × (number of permutations)
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_ordering;
  switch (topology.cell_type())
  {
  case (mesh::CellType::point):
    cell_ordering = empty_ordering(topology);
    break;
  case (mesh::CellType::interval):
    cell_ordering = empty_ordering(topology);
    break;
  case (mesh::CellType::triangle):
    cell_ordering = compute_ordering_triangle(topology);
    break;
  case (mesh::CellType::tetrahedron):
    cell_ordering = compute_ordering_tetrahedron(topology);
    break;
  case (mesh::CellType::quadrilateral):
    cell_ordering = compute_ordering_quadrilateral(topology);
    break;
  case (mesh::CellType::hexahedron):
    cell_ordering = compute_ordering_hexahedron(topology);
    break;
  default:
    // The switch should exit before this is reached
    throw std::runtime_error("Unrecognised cell type.");
  }

  // Build permutations. Each row of this represent the rotation or
  // reflection of a mesh entity Will have shape (number of
  // permutations) × (number of dofs on reference) where (number of
  // permutations) = (num_edges) for 2D cells and (num_edges + 2*num_faces)
  // for 3D cells
  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      permutations = dof_layout.base_permutations();
  const int pcols = permutations.cols();

  // Compute permutations on each cell
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p(
      cell_ordering.rows(), pcols);
  for (int i = 0; i < pcols; ++i)
    p.col(i) = i;

  // For each cell
  std::vector<int> temp(pcols);
  for (int cell = 0; cell < cell_ordering.rows(); ++cell)
  {
    // For each permutation in permutations
    for (int i = 0; i < cell_ordering.cols(); ++i)
    {
      // cell_ordering(cell, i) says how many times this permutation
      // should be applied
      for (int j = 0; j < cell_ordering(cell, i); ++j)
      {
        // This must be inside the loop as p changes after each
        // permutation
        for (int k = 0; k < pcols; ++k)
          temp[k] = p(cell, k);
        for (int k = 0; k < pcols; ++k)
          p(cell, permutations(i, k)) = temp[k];
      }
    }
  }

  return p;
}
//-----------------------------------------------------------------------------
