// Copyright (C) 2013-2026 Johan Hake, Jan Blechta, Garth N. Wells and Paul T.
// Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateElement.h"
#include "DofMap.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include "functionspace_factory.h"
#include <basix/finite-element.h>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

/// @file interpolate_geometry.h
/// @brief Functions for interpolating mesh geometry.

namespace dolfinx::fem
{
/// @brief Take an existing mesh and create a new mesh with its geometry
/// interpolated into a new coordinate element.
///
/// The topology is shared between `mesh` and the returned mesh.
///
/// Useful for creating a higher-order mesh from a lower-order one for
/// computation, or vice-versa, for IO.
///
/// @param[in] mesh Input mesh.
/// @param[in] new_cmap Coordinate element for the new geometry.
/// @param[in] reorder_fn Optional graph reordering function applied to
/// the new geometry dofmap.
/// @return A new mesh sharing the topology of `mesh` and with a
/// geometry described by `new_cmap`.
template <std::floating_point T>
mesh::Mesh<T> interpolate_geometry(
    std::shared_ptr<mesh::Mesh<T>> mesh, const CoordinateElement<T>& new_cmap,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn = nullptr)
{
  assert(mesh);
  const CoordinateElement<T>& old_cmap = mesh->geometry().cmaps().front();
  if (new_cmap.cell_shape() != old_cmap.cell_shape())
  {
    throw std::invalid_argument(
        "Cell shape of new coordinate element must match input mesh.");
  }

  const int gdim = mesh->geometry().dim();

  // Build a vector-valued Lagrange FiniteElement from the coordinate element.
  basix::FiniteElement<T> b_element = basix::create_element<T>(
      basix::element::family::P,
      mesh::cell_type_to_basix_type(new_cmap.cell_shape()), new_cmap.degree(),
      new_cmap.variant(), basix::element::dpc_variant::unset, false);
  auto element = std::make_shared<const FiniteElement<T>>(
      b_element, std::vector<std::size_t>{static_cast<std::size_t>(gdim)});

  FunctionSpace<T> V = create_functionspace(mesh, element, reorder_fn);

  // Tabulate physical coordinates of the new geometry dofs.
  std::vector<T> x_new = V.tabulate_dof_coordinates(false);

  // Pull the geometry dofmap and index map from V.
  std::shared_ptr<const DofMap> dm = V.dofmap();
  assert(dm);
  std::shared_ptr<const common::IndexMap> new_imap = dm->index_map;
  assert(new_imap);

  auto map_view = dm->map();
  std::vector<std::int32_t> dofmap_flat(
      map_view.data_handle(), map_view.data_handle() + map_view.size());

  // Build input_global_indices as the local-to-global of the new geometry
  // dofs.
  const std::int32_t num_nodes
      = new_imap->size_local() + new_imap->num_ghosts();
  std::vector<std::int32_t> local(num_nodes);
  std::iota(local.begin(), local.end(), 0);
  std::vector<std::int64_t> igi(num_nodes);
  new_imap->local_to_global(local, igi);

  mesh::Geometry<T> geometry(
      new_imap, std::vector<std::vector<std::int32_t>>{std::move(dofmap_flat)},
      std::vector<CoordinateElement<T>>{new_cmap}, std::move(x_new), gdim,
      std::move(igi));

  return mesh::Mesh<T>(mesh->comm(), mesh->topology(), std::move(geometry));
}
} // namespace dolfinx::fem
