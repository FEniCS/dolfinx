// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPI.h"
#include <mpi.h>

namespace dolfinx::common
{
class IndexMap;

/// @brief Neighbourhood communicators for the communication pattern of
/// an IndexMap.
///
/// Holds two MPI distributed-graph communicators spanning the ranks of
/// IndexMap::comm(): one for sending owned data to the ranks that ghost
/// it (owner to ghost), and its reverse for sending ghost data to the
/// owning ranks (ghost to owner). The communicators are created once,
/// here, and shared by every common::Scatterer (and hence every
/// la::Vector) built on the same IndexMap.
///
/// Construction and destruction are collective, since communicators are
/// created and freed. The class is move-only.
class NeighbourhoodComms
{
public:
  /// @brief Create the neighbourhood communicators for an index map.
  ///
  /// @note Collective on `map.comm()`.
  ///
  /// @param[in] map Index map that describes the communication pattern.
  explicit NeighbourhoodComms(const IndexMap& map);

  // Copy constructor (deleted)
  NeighbourhoodComms(const NeighbourhoodComms& comms) = delete;

  /// Move constructor
  NeighbourhoodComms(NeighbourhoodComms&& comms) = default;

  /// Destructor
  ///
  /// @note Collective, since the communicators are freed.
  ~NeighbourhoodComms() = default;

  // Copy assignment (deleted)
  NeighbourhoodComms& operator=(const NeighbourhoodComms& comms) = delete;

  /// Move assignment
  NeighbourhoodComms& operator=(NeighbourhoodComms&& comms) = default;

  /// @brief Communicator for sending owned data to the ranks that ghost
  /// it.
  ///
  /// The graph has in-edges from IndexMap::src() and out-edges to
  /// IndexMap::dest(), so a neighbourhood collective on it sends to
  /// `dest()` and receives from `src()`.
  ///
  /// @return Owner-to-ghost neighbourhood communicator.
  MPI_Comm owner_to_ghost() const noexcept;

  /// @brief Communicator for sending ghost data to the owning ranks.
  ///
  /// The reverse graph of ::owner_to_ghost: in-edges from
  /// IndexMap::dest() and out-edges to IndexMap::src().
  ///
  /// @return Ghost-to-owner neighbourhood communicator.
  MPI_Comm ghost_to_owner() const noexcept;

private:
  // In-edges from src, out-edges to dest
  dolfinx::MPI::Comm _owner_to_ghost;

  // In-edges from dest, out-edges to src
  dolfinx::MPI::Comm _ghost_to_owner;
};
} // namespace dolfinx::common
