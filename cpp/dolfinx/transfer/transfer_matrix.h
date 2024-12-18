// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <vector>

#include "dolfinx/common/IndexMap.h"
#include "dolfinx/fem/FunctionSpace.h"
#include "dolfinx/la/SparsityPattern.h"
#include "dolfinx/la/utils.h"
#include "dolfinx/mesh/Mesh.h"

namespace dolfinx::transfer
{

template <std::floating_point T>
std::vector<std::int64_t>
inclusion_mapping(const dolfinx::mesh::Mesh<T>& mesh_from,
                  const dolfinx::mesh::Mesh<T>& mesh_to)
{

  const common::IndexMap& im_from = *mesh_from.topology()->index_map(0);
  const common::IndexMap& im_to = *mesh_to.topology()->index_map(0);

  std::vector<std::int64_t> map(im_from.size_global(), -1);

  std::span<const T> x_from = mesh_from.geometry().x();
  std::span<const T> x_to = mesh_to.geometry().x();

  auto build_global_to_local = [&](const auto& im)
  {
    return [&](std::int32_t idx)
    {
      std::array<std::int64_t, 1> tmp;
      im.local_to_global(std::vector<std::int32_t>{idx}, tmp);
      return tmp[0];
    };
  };

  auto to_global_to = build_global_to_local(im_to);
  auto to_global_from = build_global_to_local(im_from);

  for (std::int32_t i = 0; i < im_from.size_local(); i++)
  {
    std::ranges::subrange vertex_from(std::next(x_from.begin(), 3 * i),
                                      std::next(x_from.begin(), 3 * (i + 1)));
    for (std::int64_t j = 0; j < im_to.size_local() + im_to.num_ghosts(); j++)
    {
      std::ranges::subrange vertex_to(std::next(x_to.begin(), 3 * j),
                                      std::next(x_to.begin(), 3 * (j + 1)));

      if (std::ranges::equal(
              vertex_from, vertex_to, [](T a, T b)
              { return std::abs(a - b) <= std::numeric_limits<T>::epsilon(); }))
      {
        map[to_global_from(i)] = to_global_to(j);
        break;
      }
    }
  }

  if (dolfinx::MPI::size(mesh_to.comm()) == 1)
  {
    // no communication required
    assert(std::ranges::all_of(map, [](auto e) { return e >= 0; }));
    return map;
  }

  // map holds at this point for every original local index the corresponding
  // mapped global index. All other entries are still -1, but are available on
  // other processes.
  std::vector<std::int64_t> result(map.size(), -1);
  MPI_Allreduce(map.data(), result.data(), map.size(),
                dolfinx::MPI::mpi_type<std::int64_t>(), MPI_MAX,
                mesh_from.comm());

  assert(std::ranges::all_of(result, [](auto e) { return e >= 0; }));
  return result;
}

template <std::floating_point T>
la::SparsityPattern
create_sparsity_pattern(const dolfinx::fem::FunctionSpace<T>& V_from,
                        const dolfinx::fem::FunctionSpace<T>& V_to,
                        const std::vector<std::int64_t>& inclusion_map)
{
  // TODO: P1 and which elements supported?
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
  // TODO: P1 and which elements supported?
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
                std::vector<double>{weight(distance)});
      }
    }
  }
}

} // namespace dolfinx::transfer