// Copyright (C) 2022-2026 Igor Baratta, Garth N. Wells and Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPI.h"
#include <cstdint>
#include <mpi.h>
#include <span>
#include <vector>

namespace dolfinx::common
{
class IndexMap;

/// @brief The MPI communication pattern of a common::IndexMap.
///
/// A ScatterPattern owns the two neighbourhood communicators over which
/// data associated with an IndexMap is scattered, together with the
/// neighbourhood sizes, displacements and index permutations. All of
/// this is a function of the IndexMap alone, so a pattern is built once
/// per IndexMap (see IndexMap::scatter_pattern) and shared by every
/// common::Scatterer built from that map, whatever the block size.
///
/// Sizes, displacements and indices are for a block size of one. A
/// Scatterer scales and expands them for its own block size.
class ScatterPattern
{
public:
  /// @brief Build the communication pattern of an index map.
  ///
  /// @note Collective on `map.comm()`.
  ///
  /// @param[in] map Index map that describes the parallel layout of
  /// data.
  explicit ScatterPattern(const IndexMap& map);

  // Copy constructor (deleted)
  ScatterPattern(const ScatterPattern& pattern) = delete;

  /// Move constructor
  ScatterPattern(ScatterPattern&& pattern) = default;

  /// Destructor
  ~ScatterPattern() = default;

  // Copy assignment (deleted)
  ScatterPattern& operator=(const ScatterPattern& pattern) = delete;

  /// Move assignment
  ScatterPattern& operator=(ScatterPattern&& pattern) = default;

  /// @brief Communicator on which owners send to the ranks that ghost
  /// their indices.
  /// @return Neighbourhood communicator, `MPI_COMM_NULL` on one rank.
  MPI_Comm comm0() const noexcept { return _comm0.comm(); }

  /// @brief Communicator on which ghosting ranks send to the owners of
  /// the ghosted indices.
  /// @return Neighbourhood communicator, `MPI_COMM_NULL` on one rank.
  MPI_Comm comm1() const noexcept { return _comm1.comm(); }

  /// @brief Number of owned indices shared with each rank in dest(),
  /// for a block size of one.
  /// @return Sizes, one per neighbour.
  std::span<const int> sizes_local() const noexcept { return _sizes_local; }

  /// @brief Displacements into the buffer of owned shared indices, for
  /// a block size of one.
  /// @return Displacements, of size `dest().size() + 1`.
  std::span<const int> displs_local() const noexcept { return _displs_local; }

  /// @brief Number of ghost indices owned by each rank in src(), for a
  /// block size of one.
  /// @return Sizes, one per neighbour.
  std::span<const int> sizes_remote() const noexcept { return _sizes_remote; }

  /// @brief Displacements into the buffer of ghost indices, for a block
  /// size of one.
  /// @return Displacements, of size `src().size() + 1`.
  std::span<const int> displs_remote() const noexcept { return _displs_remote; }

  /// @brief Permutation that sorts the index map's ghosts by owning
  /// rank, for a block size of one.
  /// @return Permutation of the ghost indices.
  std::span<const std::int32_t> perm() const noexcept { return _perm; }

  /// @brief Owned indices that are ghosted on other ranks, in local
  /// numbering and grouped by neighbouring rank, for a block size of
  /// one.
  /// @return Local indices.
  std::span<const std::int32_t> local_indices() const noexcept
  {
    return _local_inds;
  }

private:
  // Communicator where the source ranks own the indices in the callers
  // halo, and the destination ranks 'ghost' indices owned by the
  // caller. I.e.,
  // - in-edges (src) are from ranks that own my ghosts
  // - out-edges (dest) go to ranks that 'ghost' my owned indices
  dolfinx::MPI::Comm _comm0{MPI_COMM_NULL};

  // Communicator where the source ranks have ghost indices that are
  // owned by the caller, and the destination ranks are the owners of
  // indices in the callers halo region. I.e.,
  // - in-edges (src) are from ranks that 'ghost' my owned indices
  // - out-edges (dest) are to the owning ranks of my ghost indices
  dolfinx::MPI::Comm _comm1{MPI_COMM_NULL};

  // Number of remote indices (ghosts) for each neighbour process
  std::vector<int> _sizes_remote;

  // Displacements of remote data for MPI scatter and gather
  std::vector<int> _displs_remote;

  // Number of local shared indices per neighbour process
  std::vector<int> _sizes_local;

  // Displacements of local data for MPI scatter and gather
  std::vector<int> _displs_local;

  // Permutation that sorts the ghost indices by owning rank
  std::vector<std::int32_t> _perm;

  // Owned indices that are ghosted elsewhere, grouped by neighbour
  std::vector<std::int32_t> _local_inds;
};

/// @brief Build the communication pattern of an index map.
///
/// The returned pattern is owned by the caller, and is distinct from the
/// one that IndexMap::scatter_pattern shares. Use this to give a
/// common::Scatterer its own pair of neighbourhood communicators, so
/// that its scatters carry no ordering requirement against scatters on
/// other scatterers over the same map.
///
/// @note Collective on `map.comm()`. Creating a pattern creates two
/// neighbourhood communicators, a limited resource; prefer
/// IndexMap::scatter_pattern unless a private pattern is needed.
///
/// @param[in] map Index map that describes the parallel layout of data.
/// @return Communication pattern of `map`.
ScatterPattern create_scatter_pattern(const IndexMap& map);

} // namespace dolfinx::common
