// Copyright (C) 2008-2026 Anders Logg, Ola Skavhaug, Garth N. Wells and Jørgen
// S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dofmapbuilder.h"
#include <array>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/mesh/ElementDofLayout.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/dofmapbuilder.h>
#include <memory>
#include <mpi.h>
#include <vector>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::tuple<common::IndexMap, int, std::vector<std::vector<std::int32_t>>>
fem::build_dofmap_data(
    MPI_Comm comm, const mesh::Topology& topology,
    const std::vector<mesh::ElementDofLayout>& element_dof_layouts,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  return mesh::build_dofmap_data(comm, topology, element_dof_layouts,
                                 reorder_fn);
}
//-----------------------------------------------------------------------------
fem::DofMap
fem::build_real_element_dofmap(const mesh::Topology& topology,
                               const mesh::ElementDofLayout& dof_layout)
{
  // We select the process that owns cell 0 to have all dofs and all other
  // processes ghost them
  std::shared_ptr<const dolfinx::common::IndexMap> cell_map
      = topology.index_map(topology.dim());
  const int is_owner
      = (cell_map->local_range()[0] == 0) && cell_map->size_local() > 0;

  std::int32_t num_dofs = is_owner ? 1 : 0;
  std::int32_t num_ghosts = is_owner ? 0 : 1;
  std::vector<std::int64_t> ghosts(num_ghosts, 0);
  ghosts.reserve(1);

  // Send owning rank to all processes so that they can set owner of ghost dofs
  int rank = dolfinx::MPI::rank(topology.comm());
  std::array<int, 2> send_owner_pair = {is_owner, rank};
  std::array<int, 2> recv_owner_pair;
  MPI_Allreduce(&send_owner_pair, &recv_owner_pair, 1, MPI_2INT, MPI_MAXLOC,
                topology.comm());
  std::vector<int> owners(num_ghosts, recv_owner_pair[1]);
  owners.reserve(1);

  // Create index map
  auto imap = std::make_shared<const dolfinx::common::IndexMap>(
      topology.comm(), num_dofs, ghosts, owners);

  // Create dofmap array
  std::int32_t num_cells_on_process
      = topology.index_map(topology.dim())->size_local()
        + topology.index_map(topology.dim())->num_ghosts();

  std::vector<std::int32_t> dofmap(num_cells_on_process, 0);
  dofmap.reserve(1);
  return dolfinx::fem::DofMap(dof_layout, imap, dof_layout.block_size(), dofmap,
                              dof_layout.block_size());
};
//-----------------------------------------------------------------------------
