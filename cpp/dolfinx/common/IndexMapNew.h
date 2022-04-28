// Copyright (C) 2015-2019 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <memory>
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::common
{
// Forward declaration
class IndexMapNew;
class IndexMap;

/// TMP
IndexMap create_old(const IndexMapNew& map);

/// TMP
IndexMapNew create_new(const IndexMap& map);

/// @brief Given a vector of indices (local numbering, owned or ghost)
/// and an index map, this function returns the indices owned by this
/// process, including indices that might have been in the list of
/// indices on another processes.
/// @param[in] indices List of indices
/// @param[in] map The index map
/// @return Indices owned by the calling process
std::vector<int32_t>
compute_owned_indices(const xtl::span<const std::int32_t>& indices,
                      const IndexMapNew& map);

/// @brief Compute layout data and ghost indices for a stacked
/// (concatenated) index map, i.e. 'splice' multiple maps into one.
///
/// The input maps are concatenated, with indices in `maps` and owned by
/// the caller remaining owned by the caller. Ghost data is stored at
/// the end of the local range as normal, with the ghosts in blocks in
/// the order of the index maps in `maps`.
///
/// @note Index maps with a block size are unrolled in the data for the
/// concatenated index map.
/// @note Communication is required to compute the new ghost indices.
///
/// @param[in] maps List of (index map, block size) pairs
/// @returns The (0) global offset of a concatenated map for the calling
/// rank, (1) local offset for the owned indices of each submap in the
/// concatenated map, (2) new indices for the ghosts for each submap,
/// and (3) owner rank of each ghost entry for each submap.
std::tuple<std::int64_t, std::vector<std::int32_t>,
           std::vector<std::vector<std::int64_t>>,
           std::vector<std::vector<int>>>
stack_index_maps(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMapNew>, int>>&
        maps);

/// This class represents the distribution index arrays across
/// processes. An index array is a contiguous collection of N+1 indices
/// [0, 1, . . ., N] that are distributed across M processes. On a given
/// process, the IndexMapNew stores a portion of the index set using local
/// indices [0, 1, . . . , n], and a map from the local indices to a
/// unique global index.
class IndexMapNew
{
public:
  /// Create an non-overlapping index map with local_size owned on this
  /// process.
  ///
  /// @note Collective
  /// @param[in] comm The MPI communicator
  /// @param[in] local_size Local size of the IndexMapNew, i.e. the number
  /// of owned entries
  IndexMapNew(MPI_Comm comm, std::int32_t local_size);

  /// Create an index map with local_size owned indiced on this process
  ///
  /// @note Collective
  /// @param[in] comm The MPI communicator
  /// @param[in] local_size Local size of the IndexMapNew, i.e. the number
  /// of owned entries
  /// @param[in] ghosts The global indices of ghost entries
  /// @param[in] src_ranks Owner rank (on global communicator) of each
  /// entry in @p ghosts
  IndexMapNew(MPI_Comm comm, std::int32_t local_size,
              const xtl::span<const std::int64_t>& ghosts,
              const xtl::span<const int>& src_ranks);

  // Copy constructor
  IndexMapNew(const IndexMapNew& map) = delete;

  /// Move constructor
  IndexMapNew(IndexMapNew&& map) = default;

  /// Destructor
  ~IndexMapNew() = default;

  /// Move assignment
  IndexMapNew& operator=(IndexMapNew&& map) = default;

  // Copy assignment
  IndexMapNew& operator=(const IndexMapNew& map) = delete;

  /// Range of indices (global) owned by this process
  std::array<std::int64_t, 2> local_range() const noexcept;

  /// Number of ghost indices on this process
  std::int32_t num_ghosts() const noexcept;

  /// Number of indices owned by on this process
  std::int32_t size_local() const noexcept;

  /// Number indices across communicator
  std::int64_t size_global() const noexcept;

  /// Local-to-global map for ghosts (local indexing beyond end of local
  /// range)
  const std::vector<std::int64_t>& ghosts() const noexcept;

  /// Return the MPI communicator used to create the index map
  /// @return Communicator
  MPI_Comm comm() const;

  /// Compute global indices for array of local indices
  /// @param[in] local Local indices
  /// @param[out] global The global indices
  void local_to_global(const xtl::span<const std::int32_t>& local,
                       const xtl::span<std::int64_t>& global) const;

  /// Compute local indices for array of global indices
  /// @param[in] global Global indices
  /// @param[out] local The local of the corresponding global index in 'global'.
  /// Returns -1 if the local index does not exist on this process.
  void global_to_local(const xtl::span<const std::int64_t>& global,
                       const xtl::span<std::int32_t>& local) const;

  /// Global indices
  /// @return The global index for all local indices (0, 1, 2, ...) on
  /// this process, including ghosts
  std::vector<std::int64_t> global_indices() const;

  /// TMP
  const std::vector<int>& owners() const { return _owners; }

  /// @brief Create new index map from a subset of indices in this index
  /// map.
  ///
  /// The order of the owned indices is preserved, with new map
  /// effectively a 'compressed' map.
  ///
  /// @param[in] indices Local indices in the map that should appear in
  /// the new index map. All indices must be owned, i.e. indices must be
  /// less than `this->size_local()`.
  /// @pre `indices` must be sorted and contain no duplicates.
  /// @return The (i) new index map and (ii) a map from the ghost
  /// position in the new map to the ghost position in the original
  /// (this) map
  std::pair<IndexMapNew, std::vector<std::int32_t>>
  create_submap_new(const xtl::span<const std::int32_t>& indices) const;

private:
  // Range of indices (global) owned by this process
  std::array<std::int64_t, 2> _local_range;

  // Number indices across communicator
  std::int64_t _size_global;

  // MPI communicator (duplicated of 'input' communicator)
  dolfinx::MPI::Comm _comm;

  // Local-to-global map for ghost indices
  std::vector<std::int64_t> _ghosts;

  // Local-to-global map for ghost indices
  std::vector<int> _owners;
};

} // namespace dolfinx::common
