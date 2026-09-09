// Copyright (C) 2015-2024 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPI.h"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace dolfinx::common
{
// Forward declaration
class IndexMap;

/// Control ghost-index ordering in sub-index maps.
enum class IndexMapOrder : bool
{
  preserve = true, ///< Preserve the ordering of ghost indices
  any = false      ///< Allow arbitrary ghost-index ordering
};

/// @brief Return selected indices owned by the calling rank.
///
/// Includes locally owned entries in `indices` and entries selected as ghosts
/// on other ranks.
///
/// @note Collective
///
/// @param[in] indices Sorted unique local indices (owned or ghost) in
/// `[0, map.size_local() + map.num_ghosts())`.
/// @param[in] map The index map.
/// @pre `indices` is sorted, unique, and in range. This condition is checked
/// in Developer builds; callers must ensure it in Release builds.
/// @return Local indices owned by the calling rank.
/// @throws std::invalid_argument If the `indices` precondition is violated in
/// a Developer build.
std::vector<std::int32_t>
compute_owned_indices(std::span<const std::int32_t> indices,
                      const IndexMap& map);

/// @brief Compute layout data for a concatenated index map.
///
/// Locally owned entries remain owned by the caller. Ghost entries are
/// grouped by input map in `maps`.
///
/// @note Collective. Maps with a block size are unrolled.
///
/// @param[in] maps Non-empty pairs of index maps and positive block sizes.
/// All maps must use the same communicator.
/// @pre All ranks supply corresponding maps in the same order.
/// @return (0) Global offset on the calling rank, (1) local offsets for owned
/// entries in each map, (2) global ghost indices for each map, and (3) their
/// owner ranks.
std::tuple<std::int64_t, std::vector<std::int32_t>,
           std::vector<std::vector<std::int64_t>>,
           std::vector<std::vector<int>>>
stack_index_maps(
    const std::vector<std::pair<std::reference_wrapper<const IndexMap>, int>>&
        maps);

/// @brief Create an index map from a subset of an existing map.
///
/// @note Collective
///
/// @param[in] imap Parent map to create a new sub-map from.
/// @param[in] indices Local indices in `imap` (owned and ghost) to
/// include in the new index map.
/// @param[in] order Control the order in which ghost indices appear in
/// the new map.
/// @param[in] allow_owner_change Permit an index selected only by ghosting
/// ranks to acquire a new owner in the submap.
/// @pre `indices` contains unique local indices in range. This condition is
/// checked in Developer builds; callers must ensure it in Release builds.
/// @return (0) New index map and (1) corresponding local indices in `imap`.
/// @throws std::invalid_argument If ownership would change while
/// `allow_owner_change` is false, or if the `indices` precondition is violated
/// in a Developer build.
std::pair<IndexMap, std::vector<std::int32_t>> create_sub_index_map(
    const IndexMap& imap, std::span<const std::int32_t> indices,
    IndexMapOrder order = IndexMapOrder::any, bool allow_owner_change = false);

/// Distribution of a global index range `[0, N)` across MPI ranks.
///
/// Each rank owns a contiguous global range. Local indices in
/// `[0, size_local())` address owned entries; remaining local indices address
/// ghost entries.
class IndexMap
{
public:
  /// @brief Create a non-overlapping index map.
  ///
  /// @note Collective
  ///
  /// @param[in] comm MPI communicator that the index map is distributed
  /// across.
  /// @param[in] local_size Number of owned entries. Must be non-negative.
  /// @throws std::invalid_argument If `local_size` is negative.
  IndexMap(MPI_Comm comm, std::int32_t local_size);

  /// @brief Create an overlapping (ghosted) index map.
  ///
  /// Uses a consensus algorithm to determine ranks that ghost entries owned
  /// by the caller. Use the explicit source/destination constructor when these
  /// ranks are known.
  ///
  /// @note Collective
  ///
  /// @param[in] comm MPI communicator that the index map is distributed
  /// across.
  /// @param[in] local_size Number of owned entries. Must be non-negative.
  /// @param[in] ghosts Unique global indices of ghost entries.
  /// @param[in] owners Non-self rank (on `comm`) that owns each entry in
  /// `ghosts`.
  /// @param[in] tag Tag used in non-blocking MPI calls in the consensus
  /// algorithm.
  /// @note Use a distinct `tag` for overlapping consensus calls. All ranks in
  /// one collective call must use the same tag. An MPI barrier before and after
  /// the call is an alternative.
  /// @pre `ghosts` and `owners` have equal length; this is always checked.
  /// Ghosts must also be unique and non-negative, owners must be valid
  /// non-self ranks, and each ghost must be globally owned by its declared
  /// rank; these further conditions are checked in Developer builds only,
  /// and callers must ensure them in Release builds.
  /// @throws std::invalid_argument If `local_size` is negative, if `ghosts`
  /// and `owners` differ in length, or if another ghost data precondition
  /// is violated in a Developer build.
  IndexMap(MPI_Comm comm, std::int32_t local_size,
           std::span<const std::int64_t> ghosts, std::span<const int> owners,
           int tag = static_cast<int>(dolfinx::MPI::tag::consensus_nbx));

  /// @brief Create an overlapping (ghosted) index map.
  ///
  /// Use this constructor when source ranks (owners of the caller's ghosts)
  /// and destination ranks (ranks ghosting the caller's entries) are known.
  ///
  /// @note Collective
  ///
  /// @param[in] comm MPI communicator that the index map is distributed
  /// across.
  /// @param[in] local_size Number of owned entries. Must be non-negative.
  /// @param[in] src_dest Lists of (0) source and (1) destination ranks.
  /// Both lists must be sorted, unique and contain valid ranks. Source
  /// ranks must be exactly the unique owners of `ghosts`; destination
  /// ranks must be the ranks that ghost entries owned by the caller.
  /// @param[in] ghosts Unique global indices of ghost entries.
  /// @param[in] owners Non-self rank (on `comm`) that owns each entry in
  /// `ghosts`.
  /// @pre `ghosts` and `owners` have equal length; this is always checked.
  /// Ghosts must also be unique and non-negative, owners must be valid
  /// non-self ranks, and each ghost must be globally owned by its declared
  /// rank. For every pair of ranks `(a, b)`, `b` must be in `a`'s source
  /// list if and only if `a` is in `b`'s destination list. These further
  /// conditions are checked in Developer builds only, and callers must
  /// ensure them in Release builds.
  /// @throws std::invalid_argument If `local_size` is negative, if `ghosts`
  /// and `owners` differ in length, or if another ghost data precondition
  /// is violated in a Developer build.
  IndexMap(MPI_Comm comm, std::int32_t local_size,
           const std::array<std::vector<int>, 2>& src_dest,
           std::span<const std::int64_t> ghosts, std::span<const int> owners);

  // Copy constructor (deleted)
  IndexMap(const IndexMap& map) = delete;

  /// Move constructor
  IndexMap(IndexMap&& map) = default;

  /// Destructor
  ~IndexMap() = default;

  // Copy assignment (deleted)
  IndexMap& operator=(const IndexMap& map) = delete;

  /// Move assignment
  IndexMap& operator=(IndexMap&& map) = default;

  /// @brief Return the global range of owned indices.
  std::array<std::int64_t, 2> local_range() const noexcept;

  /// @brief Return the number of ghost indices.
  std::int32_t num_ghosts() const noexcept;

  /// @brief Return the number of owned indices.
  std::int32_t size_local() const noexcept;

  /// @brief Return the total number of indices across the communicator.
  std::int64_t size_global() const noexcept;

  /// @brief Return global indices of ghosts in local ghost-index order.
  std::span<const std::int64_t> ghosts() const noexcept;

  /// @brief Return the MPI communicator that the map is defined on.
  /// @return Communicator
  MPI_Comm comm() const;

  /// @brief Compute global indices for local indices.
  /// @param[in] local Local indices in `[0, size_local() + num_ghosts())`.
  /// @param[out] global Global indices. Must have at least the size of `local`.
  /// @throws std::invalid_argument If `global` is smaller than `local`.
  /// @throws std::out_of_range If a local index is out of range.
  void local_to_global(std::span<const std::int32_t> local,
                       std::span<std::int64_t> global) const;

  /// @brief Compute local indices for global indices.
  /// @param[in] global Global indices.
  /// @param[out] local Local indices. Must have the same size as `global`.
  /// Entries without a local index are set to -1.
  /// @throws std::invalid_argument If `global` and `local` differ in size.
  void global_to_local(std::span<const std::int64_t> global,
                       std::span<std::int32_t> local) const;

  /// @brief Return global indices for all local entries, including ghosts.
  std::vector<std::int64_t> global_indices() const;

  /// @brief Return ranks that own ghost entries.
  /// @return Owner ranks aligned with ghosts().
  std::span<const int> owners() const noexcept { return _owners; }

  /// @brief Compute sharing ranks for each local index.
  ///
  /// @note Collective
  ///
  /// @param[in] tag Tag to pass to MPI calls.
  /// @note See IndexMap(MPI_Comm, std::int32_t, std::span<const
  /// std::int64_t>, std::span<const int>, int) for tag requirements.
  /// @return (0) Sharing-rank data and (1) offsets. Ranks sharing local index
  /// `i` occupy `[offsets[i], offsets[i + 1])`.
  std::pair<std::vector<int>, std::vector<std::int32_t>> index_to_dest_ranks(
      int tag = static_cast<int>(dolfinx::MPI::tag::consensus_nbx)) const;

  /// @brief Return owned indices ghosted by another rank.
  ///
  /// @note Collective
  ///
  /// @return Sorted unique local indices.
  std::vector<std::int32_t> shared_indices() const;

  /// @brief Return sorted unique ranks that own the caller's ghosts.
  std::span<const int> src() const noexcept;

  /// @brief Return sorted unique ranks that ghost entries owned by the caller.
  std::span<const int> dest() const noexcept;

  /// @brief Count ghosts owned by each source rank.
  /// @return `weight[i]` is the number of ghosts owned by `src()[i]`.
  std::vector<std::int32_t> weights_src() const;

  /// @brief Count entries ghosted by each destination rank.
  ///
  /// @note Collective
  ///
  /// @return `weight[i]` is the number of entries ghosted by `dest()[i]`.
  std::vector<std::int32_t> weights_dest() const;

  /// @brief Return destination and source ranks in the caller's split group.
  ///
  /// @note Collective on comm().
  ///
  /// @param[in] split_type Type passed to MPI_Comm_split_type.
  /// @return (0) destination and (1) source ranks in the split group. Ranks
  /// are numbered on comm().
  std::array<std::vector<int>, 2> rank_type(int split_type) const;

private:
  // Global range of owned indices
  std::array<std::int64_t, 2> _local_range;

  // Global number of indices
  std::int64_t _size_global;

  // Map communicator
  dolfinx::MPI::Comm _comm;

  // Global ghost indices
  std::vector<std::int64_t> _ghosts;

  // Owner ranks for ghosts
  std::vector<int> _owners;

  // Ranks that own ghosts
  std::vector<int> _src;

  // Ranks that ghost owned entries
  std::vector<int> _dest;
};

} // namespace dolfinx::common
