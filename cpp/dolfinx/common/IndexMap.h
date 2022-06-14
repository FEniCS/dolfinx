// Copyright (C) 2015-2022 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "IndexMap.h"
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <memory>
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::common
{
// Forward declaration
class IndexMap;

/// @brief Given a vector of indices (local numbering, owned or ghost)
/// and an index map, this function returns the indices owned by this
/// process, including indices that might have been in the list of
/// indices on another processes.
/// @param[in] indices List of indices
/// @param[in] map The index map
/// @return Indices owned by the calling process
std::vector<int32_t>
compute_owned_indices(const xtl::span<const std::int32_t>& indices,
                      const IndexMap& map);

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
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

/// This class represents the distribution index arrays across
/// processes. An index array is a contiguous collection of N+1 indices
/// [0, 1, . . ., N] that are distributed across M processes. On a given
/// process, the IndexMap stores a portion of the index set using local
/// indices [0, 1, . . . , n], and a map from the local indices to a
/// unique global index.
class IndexMap
{
public:
  /// @brief Create an non-overlapping index map.
  ///
  /// @note Collective
  ///
  /// @param[in] comm The MPI communicator
  /// @param[in] local_size Local size of the index map, i.e. the number
  /// of owned entries
  IndexMap(MPI_Comm comm, std::int32_t local_size);

  /// @brief Create an overlapping (ghosted) index map.
  ///
  /// This constructor uses a 'consensus' algorithm to determine the
  /// ranks that ghost indices that are owned by the caller. This
  /// requires non-trivial MPI communication. If the ranks that ghost
  /// indices owned by the caller are known, it more efficient to use
  /// the constructor that takes these ranks as an argument.
  ///
  /// @note Collective
  ///
  /// @param[in] comm The MPI communicator
  /// @param[in] local_size Local size of the index map, i.e. the number
  /// of owned entries
  /// @param[in] ghosts The global indices of ghost entries
  /// @param[in] owners Owner rank (on global communicator) of each
  /// entry in `ghosts`
  IndexMap(MPI_Comm comm, std::int32_t local_size,
           const xtl::span<const std::int64_t>& ghosts,
           const xtl::span<const int>& owners);

  /// @brief Create an overlapping (ghosted) index map.
  ///
  /// This constructor is optimised for the case where the 'source'
  /// (ranks that own indices ghosted by the caller) and 'destination'
  /// ranks (ranks that ghost indices owned by the caller) are already
  /// available. It allows the complex computation of the destination
  /// ranks from `owners`.
  ///
  /// @note Collective
  ///
  /// @param[in] comm The MPI communicator
  /// @param[in] local_size Local size of the index map, i.e. the number
  /// @param[in] src_dest Lists of [0] src and [1] dest ranks. The list
  /// in each must be sorted and not contain duplicates. `src` ranks are
  /// owners of the indices in `ghosts`. `dest` ranks are the rank that
  /// ghost indices owned by the caller.
  /// @param[in] ghosts The global indices of ghost entries
  /// @param[in] owners Owner rank (on global communicator) of each entry
  /// in `ghosts`
  IndexMap(MPI_Comm comm, std::int32_t local_size,
           const std::array<std::vector<int>, 2>& src_dest,
           const xtl::span<const std::int64_t>& ghosts,
           const xtl::span<const int>& owners);

  // Copy constructor
  IndexMap(const IndexMap& map) = delete;

  /// Move constructor
  IndexMap(IndexMap&& map) = default;

  /// Destructor
  ~IndexMap() = default;

  /// Move assignment
  IndexMap& operator=(IndexMap&& map) = default;

  // Copy assignment
  IndexMap& operator=(const IndexMap& map) = delete;

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

  /// @brief Return the MPI communicator that the map is defined on.
  /// @return Communicator
  MPI_Comm comm() const;

  /// @brief Compute global indices for array of local indices.
  /// @param[in] local Local indices
  /// @param[out] global The global indices
  void local_to_global(const xtl::span<const std::int32_t>& local,
                       const xtl::span<std::int64_t>& global) const;

  /// @brief Compute local indices for array of global indices
  /// @param[in] global Global indices
  /// @param[out] local The local of the corresponding global index in
  /// 'global'. Returns -1 if the local index does not exist on this
  /// process.
  void global_to_local(const xtl::span<const std::int64_t>& global,
                       const xtl::span<std::int32_t>& local) const;

  /// @brief Build list of indices with global indexing.
  /// @return The global index for all local indices (0, 1, 2, ...) on
  /// this process, including ghosts
  std::vector<std::int64_t> global_indices() const;

  /// @brief The ranks that own each ghost index.
  /// @return List of ghost owners. The owning rank of the ith ghost
  /// index is `owners()[i]`.
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
  std::pair<IndexMap, std::vector<std::int32_t>>
  create_submap(const xtl::span<const std::int32_t>& indices) const;

  /// @todo Aim to remove this function?
  ///
  /// @brief Compute map from each local (owned) index to the set of
  /// ranks that have the index as a ghost.
  /// @return shared indices
  graph::AdjacencyList<int> index_to_dest_ranks() const;

  /// @brief Build a list of owned indices that are ghosted by another
  /// rank.
  /// @return The local index of owned indices that are ghosts on other
  /// rank(s). The indicies are unique and sorted.
  std::vector<std::int32_t> shared_indices() const;

  /// @brief Ordered set of MPI ranks that own caller's ghost indices.
  ///
  /// Typically used when creating neighbourhood communicators.
  ///
  /// @return MPI ranks than own ghost indices.  The ranks are unique
  /// and sorted.
  const std::vector<int>& src() const noexcept;

  /// @brief Ordered set of MPI ranks that ghost indices owned by
  /// caller.
  ///
  /// Typically used when creating neighbourhood communicators.
  ///
  /// @return MPI ranks than own ghost indices. The ranks are unique
  /// and sorted.
  const std::vector<int>& dest() const noexcept;

  /// @brief Check if index map has overlaps (ghosts on any rank).
  ///
  /// The return value of this function is determined by which
  /// constructor was used to create the index map.
  ///
  /// @return True if index map has overlaps on any ranks, otherwise
  /// false.
  bool overlapped() const noexcept;

private:
  // Range of indices (global) owned by this process
  std::array<std::int64_t, 2> _local_range;

  // Number indices across communicator
  std::int64_t _size_global;

  // MPI communicator that map is defined on
  dolfinx::MPI::Comm _comm;

  // Local-to-global map for ghost indices
  std::vector<std::int64_t> _ghosts;

  // Local-to-global map for ghost indices
  std::vector<int> _owners;

  // Set of ranks that own ghosts
  std::vector<int> _src;

  // Set of ranks ghost owned indices
  std::vector<int> _dest;

  // True if map has overlaps (ghosts)
  bool _overlapping;
};
} // namespace dolfinx::common
