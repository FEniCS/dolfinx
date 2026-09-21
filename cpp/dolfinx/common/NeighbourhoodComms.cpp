// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "NeighbourhoodComms.h"
#include "IndexMap.h"
#include "MPI.h"
#include <mpi.h>
#include <span>

using namespace dolfinx;
using namespace dolfinx::common;

namespace
{
/// Create a dist-graph communicator on `comm` with in-edges from
/// `sources` and out-edges to `destinations`
dolfinx::MPI::Comm create_graph_comm(MPI_Comm comm,
                                     std::span<const int> sources,
                                     std::span<const int> destinations)
{
  MPI_Comm graph_comm;
  int ierr = MPI_Dist_graph_create_adjacent(
      comm, sources.size(), sources.data(), MPI_UNWEIGHTED, destinations.size(),
      destinations.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &graph_comm);
  dolfinx::MPI::check_error(comm, ierr);
  return dolfinx::MPI::Comm(graph_comm, false);
}
} // namespace

//-----------------------------------------------------------------------------
NeighbourhoodComms::NeighbourhoodComms(const IndexMap& map)
    : _owner_to_ghost(create_graph_comm(map.comm(), map.src(), map.dest())),
      _ghost_to_owner(create_graph_comm(map.comm(), map.dest(), map.src()))
{
}
//-----------------------------------------------------------------------------
MPI_Comm NeighbourhoodComms::owner_to_ghost() const noexcept
{
  return _owner_to_ghost.comm();
}
//-----------------------------------------------------------------------------
MPI_Comm NeighbourhoodComms::ghost_to_owner() const noexcept
{
  return _ghost_to_owner.comm();
}
//-----------------------------------------------------------------------------
