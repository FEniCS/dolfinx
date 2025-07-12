// Copyright (C) 2015-2024 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "IndexMap.h"
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::common
{
// Forward declaration
class IndexMap;

/// Enum to control preservation of ghost index ordering in
/// sub-IndexMaps.
enum class IndexMapOrder : bool
{
  preserve = true, ///< Preserve the ordering of ghost indices
  any = false      ///< Allow arbitrary ordering of ghost indices in sub-maps
};

/// @brief Given a sorted vector of indices (local numbering, owned or
/// ghost) and an index map, this function returns the indices owned by
/// this process, including indices that might have been in the list of
/// indices on another processes.
/// @param[in] indices List of indices
/// @param[in] map The index map
/// @return Indices owned by the calling process
std::vector<int32_t>
compute_owned_indices(std::span<const std::int32_t> indices,
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
    const std::vector<std::pair<std::reference_wrapper<const IndexMap>, int>>&
        maps);

/// @brief Create a new index map from a subset of indices in an
/// existing index map.
///
/// @param[in] imap Parent map to create a new sub-map from.
/// @param[in] indices Local indices in `imap` (owned and ghost) to
/// include in the new index map.
/// @param[in] order Control the order in which ghost indices appear in
/// the new map.
/// @param[in] allow_owner_change If `true`, indices that are not
/// included in `indices` by their owning process can be included in
/// `indices` by processes that ghost the indices to be included in the
/// new submap. These indices will be owned by one of the sharing
/// processes in the submap. If `false`, and exception is raised if an
/// index is included by a sharing process and not by the owning
/// process.
/// @return The (i) new index map and (ii) a map from local indices in
/// the submap to local indices in the original (this) map.
/// @pre `indices` must be sorted and must not contain duplicates.
std::pair<IndexMap, std::vector<std::int32_t>> create_sub_index_map(
    const IndexMap& imap, std::span<const std::int32_t> indices,
    IndexMapOrder order = IndexMapOrder::any, bool allow_owner_change = false);

/// @brief Holder for statistics on an IndexMap.
///
/// IndexMap statistics provide an overview of the parallel
/// communication pattern for MPI operations based on the IndexMap.
struct IndexMapStats
{
private:
  struct minmax
  {
    std::int64_t min; ///< Minimum
    std::int64_t max; ///< Maximum
  };

public:
  // 0.
  std::int64_t num_nodes; ///< Number of nodes (MPI ranks)

  // 1. Number of node (rank) edges
  minmax out_edges; ///< Min/max number of out edges across nodes (ranks)
  minmax in_edges;  ///< Min/max number of in edges across nodes (ranks)

  // 2a. Node (rank) aggregate edges by dest/src type
  minmax out_edges_local; ///< Min/max number of out edges across nodes (ranks)
                          ///< to shared memory ranks
  minmax in_edges_local;  ///< Min/max number of out edges across nodes (ranks)
                          ///< from shared memory ranks
  // 2b.
  minmax out_edges_remote; ///< Min/max number of out edges across nodes (ranks)
                           ///< to remote memory ranks
  minmax in_edges_remote;  ///< Min/max number of edges across nodes (ranks)
                           ///< from remote memory ranks

  // 3a. Aggregate node weights
  minmax out_node_weight; ///< Min/max out weight across nodes (ranks).
  minmax in_node_weight;  ///< Min/max in weight across nodes (ranks).

  // 3b. Aggregate node weights (by dest/src type)
  minmax out_node_weight_local; ///< Min/max edge out weight to shared
                                ///< memory rank.
  // minmax in_node_weight_local; ///< Min/max edge out weight to shared
  //                               ///< memory rank.
  minmax out_node_weight_remote; ///< Min/max edge out weight to remote memory
                                 ///< rank.
  minmax in_node_weight_remote;  ///< Min/max edge in weight from
                                 ///< remote memory rank.

  // 4. Edge weights
  minmax out_edge_weight_local;  ///< Min/max out edge weight.
  minmax out_edge_weight_remote; ///< Min/max out edge weight.

  // B1. Mean number of node (rank) edges
  std::size_t edges_mean; ///< Mean number of node (rank) out(in) edges.

  // B2. Mean number of node (rank) local/remote edges
  std::size_t
      edges_local_mean; ///< Mean number of node (rank) local out(in) edges.
  std::size_t
      edges_remote_mean; ///< Mean number of node (rank) remote out(in) edges.

  // B3a.
  std::size_t node_weight_mean; ///< Mean edge weight. This is proportional to
                                ///< the mean message size.
  // B3b.
  std::size_t
      edge_weight_local_mean; ///< Mean edge weight for shared memory
                              ///< edges. This is proportional to the
                              ///< mean shared memory (local) message size.
  std::size_t
      edge_weight_remote_mean; ///< Mean edge weight. This is proportional to
                               ///< the mean remote memory message size.
};

/// This class represents the distribution index arrays across
/// processes. An index array is a contiguous collection of `N+1`
/// indices `[0, 1, . . ., N]` that are distributed across `M`
/// processes. On a given process, the IndexMap stores a portion of the
/// index set using local indices `[0, 1, . . . , n]`, and a map from
/// the local indices to a unique global index.
class IndexMap
{
public:
  /// @brief Create an non-overlapping index map.
  ///
  /// @note Collective
  ///
  /// @param[in] comm MPI communicator that the index map is distributed
  /// across.
  /// @param[in] local_size Local size of the index map, i.e. the number
  /// of owned entries.
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
  /// @param[in] comm MPI communicator that the index map is distributed
  /// across.
  /// @param[in] local_size Local size of the index map, i.e. the number
  /// of owned entries
  /// @param[in] ghosts The global indices of ghost entries
  /// @param[in] owners Owner rank (on `comm`) of each entry in `ghosts`
  /// @param[in] tag Tag used in non-blocking MPI calls in the consensus
  /// algorithm.
  /// @note A tag can sometimes be required when there are a series of
  /// calls to this constructor, or other functions that call the
  /// consensus algorithm, that are close together. In cases where this
  /// constructor is called a second time on rank and another rank has
  /// not completed its first consensus algorithm call, communications
  /// can be corrupted if each collective call of this constructor does
  /// not have its own `tag` value. Each collective call to this
  /// constructor must use the same `tag` value.  An alternative to
  /// passing a tag is to have an implicit or explicit MPI barrier
  /// before and after the call to this constructor.
  IndexMap(MPI_Comm comm, std::int32_t local_size,
           std::span<const std::int64_t> ghosts, std::span<const int> owners,
           int tag = static_cast<int>(dolfinx::MPI::tag::consensus_nbx));

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
  /// @param[in] comm MPI communicator that the index map is distributed
  /// across.
  /// @param[in] local_size Local size of the index map, i.e. the number
  /// @param[in] src_dest Lists of [0] src and [1] dest ranks. The list
  /// in each must be sorted and not contain duplicates. `src` ranks are
  /// owners of the indices in `ghosts`. `dest` ranks are the rank that
  /// ghost indices owned by the caller.
  /// @param[in] ghosts The global indices of ghost entries
  /// @param[in] owners Owner rank (on `comm`) of each entry in `ghosts`
  IndexMap(MPI_Comm comm, std::int32_t local_size,
           const std::array<std::vector<int>, 2>& src_dest,
           std::span<const std::int64_t> ghosts, std::span<const int> owners);

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

  /// Number of indices owned by this process
  std::int32_t size_local() const noexcept;

  /// Number indices across communicator
  std::int64_t size_global() const noexcept;

  /// Local-to-global map for ghosts (local indexing beyond end of local
  /// range)
  std::span<const std::int64_t> ghosts() const noexcept;

  /// @brief Return the MPI communicator that the map is defined on.
  /// @return Communicator
  MPI_Comm comm() const;

  /// @brief Compute global indices for array of local indices.
  /// @param[in] local Local indices
  /// @param[out] global The global indices
  void local_to_global(std::span<const std::int32_t> local,
                       std::span<std::int64_t> global) const;

  /// @brief Compute local indices for array of global indices.
  /// @param[in] global Global indices
  /// @param[out] local The local of the corresponding global index in
  /// 'global'. Returns -1 if the local index does not exist on this
  /// process.
  void global_to_local(std::span<const std::int64_t> global,
                       std::span<std::int32_t> local) const;

  /// @brief Build list of indices with global indexing.
  /// @return The global index for all local indices `(0, 1, 2, ...)` on
  /// this process, including ghosts
  std::vector<std::int64_t> global_indices() const;

  /// @brief The ranks that own each ghost index.
  /// @return List of ghost owners. The owning rank of the ith ghost
  /// index is `owners()[i]`.
  std::span<const int> owners() const { return _owners; }

  /// @todo Aim to remove this function?
  ///
  /// @brief Compute map from each local (owned) index to the set of
  /// ranks that have the index as a ghost.
  ///
  /// @note Collective
  ///
  /// @param[in] tag Tag to pass to MPI calls.
  /// @note See ::IndexMap(MPI_Comm,std::int32_t,std::span<const
  /// std::int64_t>,std::span<const int>,int) for an explanation of when
  /// `tag` is required.
  /// @return Shared indices.
  graph::AdjacencyList<int> index_to_dest_ranks(
      int tag = static_cast<int>(dolfinx::MPI::tag::consensus_nbx)) const;

  /// @brief Build a list of owned indices that are ghosted by another
  /// rank.
  /// @return The local index of owned indices that are ghosts on other
  /// rank(s). The indices are unique and sorted.
  std::vector<std::int32_t> shared_indices() const;

  /// @brief Ordered set of MPI ranks that own caller's ghost indices.
  ///
  /// Typically used when creating neighbourhood communicators.
  ///
  /// @return MPI ranks than own ghost indices.  The ranks are unique
  /// and sorted.
  std::span<const int> src() const noexcept;

  /// @brief Ordered set of MPI ranks that ghost indices owned by
  /// caller.
  ///
  /// Typically used when creating neighbourhood communicators.
  ///
  /// @return MPI ranks than own ghost indices. The ranks are unique
  /// and sorted.
  std::span<const int> dest() const noexcept;

  /// @brief Compute the number of ghost indices owned by each rank in
  /// IndexMap::src.
  ///
  /// This is a measure of the amount of data:
  ///
  /// 1. Sent from this rank to other ranks when performing a reverse
  /// (owner <- ghost) scatter.
  /// 2. Received by this rank from other ranks when performing a
  /// forward (owner -> ghost) scatter.
  ///
  /// @return A weight vector, where `weight[i]` the the number of
  /// ghost indices owned by rank IndexMap::src()`[i]`.
  std::vector<std::int32_t> weights_src() const;

  /// @brief
  /// @return
  std::vector<std::int32_t> weights_dest() const;

  /// @brief Destination and source ranks by type, e.g, ranks that are
  /// destination/source ranks for the caller and are in an common
  /// shared memory region.
  ///
  /// This function is used to group destination and source ranks by
  /// 'type'. The type is defined by the MPI `split_type`. Split types
  /// include ranks from a common shared memory region
  /// (`MPI_COMM_TYPE_SHARED`) or a common NUMA region. Splits types are
  /// listed at See
  /// https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Comm_split_type.3.html#split-types.
  ///
  /// @note Collective operation on comm();
  ///
  /// @param[in] split_type MPI split type, as used in the function
  /// `MPI_Comm_split_type`. See
  /// https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Comm_split_type.3.html#split-types.
  /// @return (0) Intersection of ranks in `split_type` and in dest(),
  /// and (1) intersection of ranks in `split_type` and in src().
  /// Returned ranks are on the comm() communicator.
  std::array<std::vector<int>, 2> rank_type(int split_type) const;

  /// @brief TODO
  IndexMapStats statistics() const;

  /// @brief Statistics of the IndexMap.
  ///
  /// Returns a JSON formatted string with min,max,mean and sd for
  /// local_size, ghost_size, the number of incoming and outgoing
  /// neighbors, and the message size between neighbors for ghost
  /// update.
  ///
  /// @note This is a collective operation, but output is only on rank
  /// 0. All other ranks return an empty string.
  ///
  /// @param detail_level Set to 1 to get a more detailed per-process
  /// message size breakdown.
  /// @return JSON formatted string.
  std::string stats(int detail_level) const;

private:
  // Range of indices (global) owned by this process
  std::array<std::int64_t, 2> _local_range;

  // Number indices across communicator
  std::int64_t _size_global;

  // MPI communicator that map is defined on
  dolfinx::MPI::Comm _comm;

  // Local-to-global map for ghost indices
  std::vector<std::int64_t> _ghosts;

  // Owning rank on _comm for the ith ghost index
  std::vector<int> _owners;

  // Set of ranks that own ghosts
  std::vector<int> _src;

  // Set of ranks ghost owned indices
  std::vector<int> _dest;
};
} // namespace dolfinx::common
