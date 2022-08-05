// Copyright (C) 2005-2022 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "vtk_utils.h"
#include "cells.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <span>
#include <tuple>

using namespace dolfinx;

namespace
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
std::tuple<std::vector<double>, std::array<std::size_t, 2>,
           std::vector<std::int64_t>, std::vector<std::uint8_t>>
tabulate_lagrange_dof_coordinates(const fem::FunctionSpace& V)
{
  auto mesh = V.mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

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
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& dofmap_x
      = mesh->geometry().dofmap();
  std::span<const double> x_g = mesh->geometry().x();
  const std::size_t num_dofs_g = cmap.dim();

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
  }
  const auto apply_dof_transformation
      = element->get_dof_transformation_function<double>();

  namespace stdex = std::experimental;
  using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using cmdspan4_t = stdex::mdspan<double, stdex::dextents<std::size_t, 4>>;

  // Tabulate basis functions at node reference coordinates
  const std::array<std::size_t, 4> phi_shape
      = cmap.tabulate_shape(0, Xshape[0]);
  std::vector<double> phi_b(
      std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi_full(phi_b.data(), phi_shape);
  cmap.tabulate(0, X, Xshape, phi_b);
  auto phi = stdex::submdspan(phi_full, 0, stdex::full_extent,
                              stdex::full_extent, 0);

  // Loop over cells and tabulate dofs
  auto map = mesh->topology().index_map(tdim);
  assert(map);
  const std::int32_t num_cells = map->size_local() + map->num_ghosts();
  std::vector<double> x_b(scalar_dofs * gdim);
  mdspan2_t x(x_b.data(), scalar_dofs, gdim);
  std::vector<double> coordinate_dofs_b(num_dofs_g * gdim);
  mdspan2_t coordinate_dofs(coordinate_dofs_b.data(), num_dofs_g, gdim);

  std::vector<double> coords(num_nodes * 3, 0.0);
  std::array<std::size_t, 2> cshape = {num_nodes, 3};
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    // Extract cell geometry
    auto dofs_x = dofmap_x.links(c);
    for (std::size_t i = 0; i < dofs_x.size(); ++i)
      for (std::size_t j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g[3 * dofs_x[i] + j];

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

  // Origina points IDs
  std::vector<std::int64_t> x_id(num_nodes);
  std::array<std::int64_t, 2> range = map_dofs->local_range();
  std::int32_t size_local = range[1] - range[0];
  std::iota(x_id.begin(), std::next(x_id.begin(), size_local), range[0]);
  const std::vector<std::int64_t>& ghosts = map_dofs->ghosts();
  std::copy(ghosts.begin(), ghosts.end(), std::next(x_id.begin(), size_local));

  // Ghosts
  std::vector<std::uint8_t> id_ghost(num_nodes, 0);
  std::fill(std::next(id_ghost.begin(), size_local), id_ghost.end(), 1);

  return {std::move(coords), cshape, std::move(x_id), std::move(id_ghost)};
}
} // namespace

//-----------------------------------------------------------------------------
std::tuple<std::vector<double>, std::array<std::size_t, 2>,
           std::vector<std::int64_t>, std::vector<std::uint8_t>,
           std::vector<std::int64_t>, std::array<std::size_t, 2>>
io::vtk_mesh_from_space(const fem::FunctionSpace& V)
{
  auto mesh = V.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  assert(V.element());
  if (V.element()->is_mixed())
    throw std::runtime_error("Can't create VTK mesh from a mixed element");

  const auto [x, xshape, x_id, x_ghost] = tabulate_lagrange_dof_coordinates(V);
  auto map = mesh->topology().index_map(tdim);
  const std::size_t num_cells = map->size_local() + map->num_ghosts();

  // Create permutation from DOLFINx dof ordering to VTK
  auto dofmap = V.dofmap();
  assert(dofmap);
  const int element_block_size = V.element()->block_size();
  const std::uint32_t num_nodes
      = V.element()->space_dimension() / element_block_size;
  const std::vector<std::uint8_t> vtkmap = io::cells::transpose(
      io::cells::perm_vtk(mesh->topology().cell_type(), num_nodes));

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
//-----------------------------------------------------------------------------
std::pair<std::vector<std::int64_t>, std::array<std::size_t, 2>>
io::extract_vtk_connectivity(const mesh::Mesh& mesh)
{
  // Get DOLFINx to VTK permutation
  // FIXME: Use better way to get number of nodes
  const graph::AdjacencyList<std::int32_t>& dofmap_x = mesh.geometry().dofmap();
  const std::size_t num_nodes = mesh.geometry().cmap().dim();
  mesh::CellType cell_type = mesh.topology().cell_type();
  std::vector vtkmap
      = io::cells::transpose(io::cells::perm_vtk(cell_type, num_nodes));

  // Extract mesh 'nodes'
  const int tdim = mesh.topology().dim();
  const std::size_t num_cells = mesh.topology().index_map(tdim)->size_local()
                                + mesh.topology().index_map(tdim)->num_ghosts();

  // Build mesh connectivity

  // Loop over cells
  std::array<std::size_t, 2> shape = {num_cells, num_nodes};
  std::vector<std::int64_t> topology(shape[0] * shape[1]);
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    // For each cell, get the 'nodes' and place in VTK order
    auto dofs_x = dofmap_x.links(c);
    for (std::size_t i = 0; i < dofs_x.size(); ++i)
      topology[c * shape[1] + i] = dofs_x[vtkmap[i]];
  }

  return {std::move(topology), shape};
}
//-----------------------------------------------------------------------------
