// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <basix/element-families.h>

#include "dolfinx/common/IndexMap.h"
#include "dolfinx/fem/FunctionSpace.h"
#include "dolfinx/la/SparsityPattern.h"
#include "dolfinx/la/utils.h"

namespace dolfinx::multigrid
{

template <std::floating_point T>
la::SparsityPattern
create_sparsity_pattern(const dolfinx::fem::FunctionSpace<T>& V_from,
                        const dolfinx::fem::FunctionSpace<T>& V_to,
                        const std::vector<std::int64_t>& inclusion_map)
{
  if (*V_from.element() != *V_to.element())
    throw std::runtime_error(
        "Transfer between different element types not supported");

  if (V_from.element()->basix_element().family() != basix::element::family::P)
    throw std::runtime_error("Only Lagrange elements supported");

  if (V_from.element()->basix_element().degree() != 1)
    throw std::runtime_error("Only piecewise linear elements supported");

  // TODO: mixed elements? value shapes? DG?

  auto mesh_from = V_from.mesh();
  auto mesh_to = V_to.mesh();
  assert(mesh_from);
  assert(mesh_to);

  MPI_Comm comm = mesh_from->comm();
  {
    // Check comms equal
    int result;
    MPI_Comm_compare(comm, mesh_to->comm(), &result);
    assert(result == MPI_CONGRUENT);
  }
  assert(mesh_from->topology()->dim() == mesh_to->topology()->dim());

  auto to_v_to_f = mesh_to->topology()->connectivity(0, 1);
  auto to_f_to_v = mesh_to->topology()->connectivity(1, 0);
  assert(to_v_to_f);
  assert(to_f_to_v);

  auto dofmap_from = V_from.dofmap();
  auto dofmap_to = V_to.dofmap();
  assert(dofmap_from);
  assert(dofmap_to);

  assert(mesh_to->topology()->index_map(0));
  assert(mesh_from->topology()->index_map(0));
  const common::IndexMap& im_to = *mesh_to->topology()->index_map(0);
  const common::IndexMap& im_from = *mesh_from->topology()->index_map(0);

  dolfinx::la::SparsityPattern sp(
      comm, {dofmap_from->index_map, dofmap_to->index_map},
      {dofmap_from->index_map_bs(), dofmap_to->index_map_bs()});

  assert(inclusion_map.size() == im_from.size_global());
  for (int dof_from_global = 0; dof_from_global < im_from.size_global();
       dof_from_global++)
  {
    std::int64_t dof_to_global = inclusion_map[dof_from_global];

    std::vector<std::int32_t> local_dof_to_v{0};
    im_to.global_to_local(std::vector<std::int64_t>{dof_to_global},
                          local_dof_to_v);

    auto local_dof_to = local_dof_to_v[0];

    bool is_remote = (local_dof_to == -1);
    bool is_ghost = local_dof_to >= im_to.size_local();
    if (is_remote || is_ghost)
      continue;

    std::vector<std::int32_t> dof_from_v{0};
    im_from.global_to_local(std::vector<std::int64_t>{dof_from_global},
                            dof_from_v);

    std::ranges::for_each(
        to_v_to_f->links(local_dof_to),
        [&](auto e)
        {
          sp.insert(
              std::vector<int32_t>(to_f_to_v->links(e).size(), dof_from_v[0]),
              to_f_to_v->links(e));
        });
  }
  sp.finalize();
  return sp;
}

template <std::floating_point T>
void assemble_transfer_matrix(la::MatSet<T> auto mat_set,
                              const dolfinx::fem::FunctionSpace<T>& V_from,
                              const dolfinx::fem::FunctionSpace<T>& V_to,
                              const std::vector<std::int64_t>& inclusion_map,
                              std::function<T(std::int32_t)> weight)
{
  if (*V_from.element() != *V_to.element())
    throw std::runtime_error(
        "Transfer between different element types not supported");

  if (V_from.element()->basix_element().family() != basix::element::family::P)
    throw std::runtime_error("Only Lagrange elements supported");

  if (V_from.element()->basix_element().degree() != 1)
    throw std::runtime_error("Only piecewise linear elements supported");

  // TODO: mixed elements? value shapes? DG?

  auto mesh_from = V_from.mesh();
  auto mesh_to = V_to.mesh();
  assert(mesh_from);

  MPI_Comm comm = mesh_from->comm();
  {
    // Check comms equal
    int result;
    MPI_Comm_compare(comm, mesh_to->comm(), &result);
    assert(result == MPI_CONGRUENT);
  }
  assert(mesh_from->topology()->dim() == mesh_to->topology()->dim());

  auto to_v_to_f = mesh_to->topology()->connectivity(0, 1);
  auto to_f_to_v = mesh_to->topology()->connectivity(1, 0);
  assert(to_v_to_f);
  assert(to_f_to_v);

  auto dofmap_from = V_from.dofmap();
  auto dofmap_to = V_to.dofmap();
  assert(dofmap_from);
  assert(dofmap_to);

  assert(mesh_to->topology()->index_map(0));
  assert(mesh_from->topology()->index_map(0));
  const common::IndexMap& im_to = *mesh_to->topology()->index_map(0);
  const common::IndexMap& im_from = *mesh_from->topology()->index_map(0);

  assert(inclusion_map.size() == im_from.size_global());

  for (int dof_from_global = 0; dof_from_global < im_from.size_global();
       dof_from_global++)
  {
    std::int64_t dof_to_global = inclusion_map[dof_from_global];

    std::vector<std::int32_t> local_dof_to_v{0};
    im_to.global_to_local(std::vector<std::int64_t>{dof_to_global},
                          local_dof_to_v);

    auto local_dof_to = local_dof_to_v[0];

    bool is_remote = (local_dof_to == -1);
    bool is_ghost = local_dof_to >= im_to.size_local();
    if (is_remote || is_ghost)
      continue;

    std::vector<std::int32_t> dof_from_v{0};
    im_from.global_to_local(std::vector<std::int64_t>{dof_from_global},
                            dof_from_v);

    for (auto e : to_v_to_f->links(local_dof_to))
    {
      for (auto n : to_f_to_v->links(e))
      {
        // For now we only support distance <= 1 -> this should be somehow
        // asserted.
        std::int32_t distance = (n == local_dof_to) ? 0 : 1;
        mat_set(std::vector<int32_t>{dof_from_v[0]}, std::vector<int32_t>{n},
                std::vector<T>{weight(distance)});
      }
    }
  }
}

} // namespace dolfinx::multigrid