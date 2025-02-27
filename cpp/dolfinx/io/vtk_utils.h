// Copyright (C) 2020-2022 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "cells.h"
#include "vtk_utils.h"
#include <algorithm>
#include <array>
#include <basix/mdspan.hpp>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <span>
#include <tuple>
#include <vector>

namespace dolfinx
{
namespace fem
{
template <std::floating_point T>
class FunctionSpace;
}

namespace mesh
{
enum class CellType;
template <std::floating_point T>
class Geometry;
} // namespace mesh

namespace io
{
namespace impl
{
/// Tabulate the coordinate for every 'node' in a Lagrange function
/// space.
/// @pre `V` must be (discontinuous) Lagrange and must not be a subspace
/// @param[in] V The function space
/// @return Mesh coordinate data
/// -# Node coordinates (shape={num_dofs, 3}) where the ith row
/// corresponds to the coordinate of the ith dof in `V` (local to
/// process)
/// -# Node coordinates shape
/// -# Unique global index for each node
/// -# ghost index for each node (0=non-ghost, 1=ghost)
template <typename T>
std::tuple<std::vector<T>, std::array<std::size_t, 2>,
           std::vector<std::int64_t>, std::vector<std::uint8_t>>
tabulate_lagrange_dof_coordinates(const fem::FunctionSpace<T>& V)
{
  auto mesh = V.mesh();
  assert(mesh);
  auto topology = mesh->topology();
  assert(topology);
  const std::size_t gdim = mesh->geometry().dim();
  const int tdim = topology->dim();

  // Get dofmap data
  auto dofmap = V.dofmap();
  assert(dofmap);
  auto map_dofs = dofmap->index_map;
  assert(map_dofs);
  const int index_map_bs = dofmap->index_map_bs();
  const int dofmap_bs = dofmap->bs();

  // Get element data
  auto element = V.element();
  assert(element);
  const int e_bs = element->block_size();
  const std::size_t scalar_dofs = element->space_dimension() / e_bs;
  const std::size_t num_nodes
      = index_map_bs * (map_dofs->size_local() + map_dofs->num_ghosts())
        / dofmap_bs;

  // Get the dof coordinates on the reference element and the  mesh
  // coordinate map
  const auto [X, Xshape] = element->interpolation_points();
  const fem::CoordinateElement<T>& cmap = mesh->geometry().cmap();

  // Prepare cell geometry
  auto dofmap_x = mesh->geometry().dofmap();
  std::span<const T> x_g = mesh->geometry().x();
  const std::size_t num_dofs_g = cmap.dim();

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  // Transformation from reference element basis function data to
  // conforming element basis function function
  auto apply_dof_transformation
      = element->template dof_transformation_fn<T>(fem::doftransform::standard);

  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

  // Tabulate basis functions at node reference coordinates
  const std::array<std::size_t, 4> phi_shape
      = cmap.tabulate_shape(0, Xshape[0]);
  std::vector<T> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi_full(phi_b.data(), phi_shape);
  cmap.tabulate(0, X, Xshape, phi_b);
  auto phi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      phi_full, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Loop over cells and tabulate dofs
  auto map = topology->index_map(tdim);
  assert(map);
  const std::int32_t num_cells = map->size_local() + map->num_ghosts();
  std::vector<T> x_b(scalar_dofs * gdim);
  mdspan2_t x(x_b.data(), scalar_dofs, gdim);
  std::vector<T> coordinate_dofs_b(num_dofs_g * gdim);
  mdspan2_t coordinate_dofs(coordinate_dofs_b.data(), num_dofs_g, gdim);

  std::vector<T> coords(num_nodes * 3, 0.0);
  std::array<std::size_t, 2> cshape = {num_nodes, 3};
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    // Extract cell geometry
    for (std::size_t i = 0; i < dofmap_x.extent(1); ++i)
      for (std::size_t j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g[3 * dofmap_x(c, i) + j];

    // Tabulate dof coordinates on cell
    cmap.push_forward(x, coordinate_dofs, phi);
    apply_dof_transformation(x_b, std::span(cell_info.data(), cell_info.size()),
                             c, x.extent(1));

    // Copy dof coordinates into vector
    auto dofs = dofmap->cell_dofs(c);
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
      std::int32_t dof = dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coords[3 * dof + j] = x(i, j);
    }
  }

  // Original point IDs
  std::vector<std::int64_t> x_id(num_nodes);
  std::array<std::int64_t, 2> range = map_dofs->local_range();
  std::int32_t size_local = range[1] - range[0];
  std::iota(x_id.begin(), std::next(x_id.begin(), size_local), range[0]);
  std::span ghosts = map_dofs->ghosts();
  std::ranges::copy(ghosts, std::next(x_id.begin(), size_local));

  // Ghosts
  std::vector<std::uint8_t> id_ghost(num_nodes, 0);
  std::fill(std::next(id_ghost.begin(), size_local), id_ghost.end(), 1);

  return {std::move(coords), cshape, std::move(x_id), std::move(id_ghost)};
}

} // namespace impl

/// @brief Given a FunctionSpace, create a topology and geometry based
/// on the dof coordinates.
///
/// @pre `V` must be a (discontinuous) Lagrange space
///
/// @param[in] V The function space
/// @returns Mesh data
/// -# node coordinates (shape={num_nodes, 3}), row-major storage
/// -# node coordinates shape
/// -# unique global ID for each node (a node that appears on more than
/// one rank will have the same global ID)
/// -# ghost index for each node (0=non-ghost, 1=ghost)
/// -# cells (shape={num_cells, nodes_per_cell)}), row-major storage
/// -# cell shape (shape={num_cells, nodes_per_cell)})
template <typename T>
std::tuple<std::vector<T>, std::array<std::size_t, 2>,
           std::vector<std::int64_t>, std::vector<std::uint8_t>,
           std::vector<std::int64_t>, std::array<std::size_t, 2>>
vtk_mesh_from_space(const fem::FunctionSpace<T>& V)
{
  auto mesh = V.mesh();
  assert(mesh);
  auto topology = mesh->topology();
  assert(topology);
  const int tdim = topology->dim();

  assert(V.element());
  if (V.element()->is_mixed())
    throw std::runtime_error("Can't create VTK mesh from a mixed element");

  const auto [x, xshape, x_id, x_ghost]
      = impl::tabulate_lagrange_dof_coordinates(V);
  auto map = topology->index_map(tdim);
  const std::size_t num_cells = map->size_local() + map->num_ghosts();

  // Create permutation from DOLFINx dof ordering to VTK
  auto dofmap = V.dofmap();
  assert(dofmap);
  const int element_block_size = V.element()->block_size();
  const std::uint32_t num_nodes
      = V.element()->space_dimension() / element_block_size;
  const std::vector<std::uint16_t> vtkmap = io::cells::transpose(
      io::cells::perm_vtk(topology->cell_type(), num_nodes));

  // Extract topology for all local cells as
  // [v0_0, ...., v0_N0, v1_0, ...., v1_N1, ....]
  std::array<std::size_t, 2> shape = {num_cells, num_nodes};
  std::vector<std::int64_t> vtk_topology(shape[0] * shape[1]);
  for (std::size_t c = 0; c < shape[0]; ++c)
  {
    auto dofs = dofmap->cell_dofs(c);
    for (std::size_t i = 0; i < dofs.size(); ++i)
      vtk_topology[c * shape[1] + i] = dofs[vtkmap[i]];
  }

  return {std::move(x),
          xshape,
          std::move(x_id),
          std::move(x_ghost),
          std::move(vtk_topology),
          shape};
}

/// @brief Extract the cell topology (connectivity) in VTK ordering for
/// all cells the mesh. The 'topology' includes higher-order 'nodes'.
///
/// The index of a 'node' corresponds to the index of DOLFINx geometry
/// 'nodes'.
///
/// @param[in] dofmap_x Geometry dofmap.
/// @param[in] cell_type Cell type.
/// @return Cell topology in VTK ordering and in term of the DOLFINx
/// geometry 'nodes'.
/// @note The indices in the return array correspond to the point
/// indices in the mesh geometry array.
/// @note Even if the indices are local (int32), VTX requires int64 as
/// local input.
std::pair<std::vector<std::int64_t>, std::array<std::size_t, 2>>
extract_vtk_connectivity(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        dofmap_x,
    mesh::CellType cell_type);

/// Get VTK cell identifier
/// @param[in] cell The cell type
/// @param[in] dim The topological dimension of the cell
/// @return The VTK cell identifier
std::int8_t get_vtk_cell_type(mesh::CellType cell, int dim);

} // namespace io
} // namespace dolfinx
