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
#include <tuple>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

using namespace dolfinx;

namespace
{
/// Tabulate the coordinate for every 'node' in a Lagrange function
/// space.
/// @param[in] V The function space
/// @return Mesh coordinate data
/// -# Node coordinates (shape={num_dofs, 3}) where the ith row
/// corresponds to the coordinate of the ith dof in `V` (local to
/// process)
/// -# Unique global index for each node
/// -# ghost index for each node (0=non-ghost, 1=ghost)
/// @pre `V` must be (discontinuous) Lagrange and must not be a subspace
std::tuple<xt::xtensor<double, 2>, std::vector<std::int64_t>,
           std::vector<std::uint8_t>>
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
  const std::int32_t num_nodes
      = index_map_bs * (map_dofs->size_local() + map_dofs->num_ghosts())
        / dofmap_bs;

  // Get the dof coordinates on the reference element and the  mesh
  // coordinate map
  const xt::xtensor<double, 2>& X = element->interpolation_points();
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& dofmap_x
      = mesh->geometry().dofmap();
  xtl::span<const double> x_g = mesh->geometry().x();
  const std::size_t num_dofs_g = cmap.dim();

  xtl::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }
  const auto apply_dof_transformation
      = element->get_dof_transformation_function<double>();

  // Tabulate basis functions at node reference coordinates
  const xt::xtensor<double, 2> phi
      = xt::view(cmap.tabulate(0, X), 0, xt::all(), xt::all(), 0);

  // Loop over cells and tabulate dofs
  auto map = mesh->topology().index_map(tdim);
  assert(map);
  const std::int32_t num_cells = map->size_local() + map->num_ghosts();
  xt::xtensor<double, 2> x = xt::zeros<double>({scalar_dofs, gdim});
  xt::xtensor<double, 2> coordinate_dofs({num_dofs_g, gdim});
  xt::xtensor<double, 2> coords = xt::zeros<double>({num_nodes, 3});
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    // Extract cell geometry
    auto dofs_x = dofmap_x.links(c);
    for (std::size_t i = 0; i < dofs_x.size(); ++i)
    {
      std::copy_n(std::next(x_g.begin(), 3 * dofs_x[i]), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Tabulate dof coordinates on cell
    cmap.push_forward(x, coordinate_dofs, phi);
    apply_dof_transformation(xtl::span(x.data(), x.size()),
                             xtl::span(cell_info.data(), cell_info.size()), c,
                             x.shape(1));

    // Copy dof coordinates into vector
    auto dofs = dofmap->cell_dofs(c);
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
      std::int32_t dof = dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coords(dof, j) = x(i, j);
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

  return {std::move(coords), std::move(x_id), std::move(id_ghost)};
}
} // namespace

//-----------------------------------------------------------------------------
std::tuple<xt::xtensor<double, 2>, std::vector<std::int64_t>,
           std::vector<std::uint8_t>, xt::xtensor<std::int32_t, 2>>
io::vtk_mesh_from_space(const fem::FunctionSpace& V)
{
  auto mesh = V.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  assert(V.element());
  if (V.element()->is_mixed())
    throw std::runtime_error("Can create VTK mesh from a mixed element");

  const auto [x, x_id, x_ghost] = tabulate_lagrange_dof_coordinates(V);
  auto map = mesh->topology().index_map(tdim);
  const std::size_t num_cells = map->size_local() + map->num_ghosts();

  // Create permutation from DOLFINx dof ordering to VTK
  auto dofmap = V.dofmap();
  assert(dofmap);
  const std::uint32_t num_nodes = dofmap->cell_dofs(0).size();
  const std::vector<std::uint8_t> vtkmap = io::cells::transpose(
      io::cells::perm_vtk(mesh->topology().cell_type(), num_nodes));

  // Extract topology for all local cells as
  // [v0_0, ...., v0_N0, v1_0, ...., v1_N1, ....]
  xt::xtensor<std::int32_t, 2> vtk_topology({num_cells, num_nodes});
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    auto dofs = dofmap->cell_dofs(c);
    for (std::size_t i = 0; i < dofs.size(); ++i)
      vtk_topology(c, i) = dofs[vtkmap[i]];
  }

  return {std::move(x), std::move(x_id), std::move(x_ghost),
          std::move(vtk_topology)};
}
//-----------------------------------------------------------------------------
xt::xtensor<std::int64_t, 2>
io::extract_vtk_connectivity(const mesh::Mesh& mesh)
{
  // Get DOLFINx to VTK permutation
  // FIXME: Use better way to get number of nodes
  const graph::AdjacencyList<std::int32_t>& dofmap_x = mesh.geometry().dofmap();
  const std::size_t num_nodes = mesh.geometry().cmap().dim();
  mesh::CellType cell_type = mesh.topology().cell_type();
  std::vector vtkmap
      = io::cells::transpose(io::cells::perm_vtk(cell_type, num_nodes));

  // TODO: Remove when when paraview issue 19433 is resolved
  // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
  if (cell_type == mesh::CellType::hexahedron and num_nodes == 27)
  {
    vtkmap = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
              22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
  }
  // Extract mesh 'nodes'
  const int tdim = mesh.topology().dim();
  const std::size_t num_cells = mesh.topology().index_map(tdim)->size_local()
                                + mesh.topology().index_map(tdim)->num_ghosts();

  // Build mesh connectivity

  // Loop over cells
  xt::xtensor<std::int64_t, 2> topology({num_cells, num_nodes});
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    // For each cell, get the 'nodes' and place in VTK order
    auto dofs_x = dofmap_x.links(c);
    for (std::size_t i = 0; i < dofs_x.size(); ++i)
      topology(c, i) = dofs_x[vtkmap[i]];
  }

  return topology;
}
//-----------------------------------------------------------------------------
