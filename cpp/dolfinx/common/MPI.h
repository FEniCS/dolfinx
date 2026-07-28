// Copyright (C) 2007-2023 Magnus Vikstrøm, Garth N. Wells and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Timer.h"
#include "local_range.h"
#include "log.h"
#include "sort.h"
#include "types.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#define MPICH_IGNORE_CXX_SEEK 1
#include <mpi.h>

/// @brief MPI support functionality
namespace dolfinx::MPI
{
/// MPI communication tags
enum class tag : int
{
  consensus_pcx = 1200,
  consensus_pex = 1201,
  consensus_nbx = 1202,
};

/// @brief A duplicate MPI communicator and manage lifetime of the
/// communicator.
class Comm
{
public:
  /// Duplicate communicator and wrap duplicate
  explicit Comm(MPI_Comm comm, bool duplicate = true);

  /// Copy constructor
  Comm(const Comm& comm) noexcept;

  /// Move constructor
  Comm(Comm&& comm) noexcept;

  // Disable copy assignment operator
  Comm& operator=(const Comm& comm) = delete;

  /// Move assignment operator
  Comm& operator=(Comm&& comm) noexcept;

  /// Destructor (frees wrapped communicator)
  ~Comm();

  /// Return the underlying MPI_Comm object
  MPI_Comm comm() const noexcept;

private:
  // MPI communicator
  MPI_Comm _comm;
};

/// Return process rank for the communicator
int rank(MPI_Comm comm);

/// Return size of the group (number of processes) associated with the
/// communicator.
int size(MPI_Comm comm);

/// @brief Check MPI error code. If the error code is not equal to
/// MPI_SUCCESS, then std::abort is called.
/// @param[in] comm MPI communicator.
/// @param[in] code Error code returned by an MPI function call.
void check_error(MPI_Comm comm, int code);

/// @brief Return which rank owns index in global range [0, N - 1]
/// (inverse of MPI::local_range).
/// @param[in] size Number of MPI ranks.
/// @param[in] index The index to determine the owning rank of.
/// @param[in] N Total number of indices.
/// @return Rank of the owning process.
constexpr int index_owner(int size, std::size_t index, std::size_t N)
{
  assert(index < N);

  // Compute number of items per rank and remainder
  const std::size_t n = N / size;
  const std::size_t r = N % size;

  if (index < r * (n + 1))
  {
    // First r ranks own n + 1 indices
    return index / (n + 1);
  }
  else
  {
    // Remaining ranks own n indices
    return r + (index - r * (n + 1)) / n;
  }
}

/// @brief Determine incoming graph edges using the PCX consensus
/// algorithm.
///
/// Given a list of outgoing edges (destination ranks) from this rank,
/// this function returns the incoming edges (source ranks) to this rank.
///
/// @note This function is for sparse communication patterns, i.e. where
/// the number of ranks that communicate with each other is relatively
/// small. It **is not** scalable as arrays the size of the communicator
/// are allocated. It implements the PCX algorithm described in
/// https://dx.doi.org/10.1145/1837853.1693476.
///
/// @note For sparse graphs, this function has \f$O(p)\f$ cost, where
/// \f$p\f$is the number of MPI ranks. It is suitable for modest MPI
/// rank counts.
///
/// @note The order of the returned ranks is not deterministic.
///
/// @note Collective
///
/// @param[in] comm MPI communicator
/// @param[in] edges Edges (ranks) from this rank (the caller).
/// @return Ranks that have defined edges from them to this rank.
std::vector<int> compute_graph_edges_pcx(MPI_Comm comm,
                                         std::span<const int> edges);

/// @brief Determine incoming graph edges using the NBX consensus
/// algorithm.
///
/// Given a list of outgoing edges (destination ranks) from this rank,
/// this function returns the incoming edges (source ranks) to this rank.
///
/// @note This function is for sparse communication patterns, i.e. where
/// the number of ranks that communicate with each other is relatively
/// small. It is scalable, i.e. no arrays the size of the communicator
/// are constructed and the communication pattern is sparse. It
/// implements the NBX algorithm presented in
/// https://dx.doi.org/10.1145/1837853.1693476.
///
/// @note The order of the returned ranks is not deterministic.
///
/// @note Collective.
///
/// @param[in] comm MPI communicator
/// @param[in] edges Edges (ranks) from this rank (the caller).
/// @pre `edges` must not contain duplicate ranks.
/// @param[in] tag Tag used in non-blocking MPI calls. A tag can be
/// required when this function is called a second time on some ranks
/// before a previous call has completed on all other ranks.
/// @return Ranks that have defined edges from them to this rank.
/// @note An alternative to passing a tag is to ensure that there is
/// an implicit or explicit barrier before and after the call to this
/// function.
std::vector<int>
compute_graph_edges_nbx(MPI_Comm comm, std::span<const int> edges,
                        int tag = static_cast<int>(tag::consensus_nbx));

/// @brief Determine incoming graph edges for two independent edge
/// sets using a single, overlapped NBX consensus round.
///
/// Equivalent to calling MPI::compute_graph_edges_nbx independently
/// for `edges0` and `edges1`, but runs both consensus rounds
/// concurrently behind a single shared barrier rather than one after
/// the other, halving the number of global barrier round-trips when
/// the two edge sets can be computed without waiting on one another.
///
/// @note `tag0` and `tag1` must differ from one another, and from any
/// tag used by another consensus round that may be in flight
/// concurrently on `comm`.
///
/// @note Collective.
///
/// @param[in] comm MPI communicator
/// @param[in] edges0 Edges (ranks) from this rank (the caller) for
/// the first edge set.
/// @param[in] tag0 Tag used for `edges0`'s messages.
/// @param[in] edges1 Edges (ranks) from this rank (the caller) for
/// the second edge set.
/// @param[in] tag1 Tag used for `edges1`'s messages.
/// @return Ranks with defined edges to this rank, for (0) `edges0`
/// and (1) `edges1`.
std::pair<std::vector<int>, std::vector<int>>
compute_graph_edges_nbx(MPI_Comm comm, std::span<const int> edges0, int tag0,
                        std::span<const int> edges1, int tag1);

/// @brief Send row data to its 'post office' rank.
///
/// `x` is a contiguous local block of a larger row-major array
/// distributed over `comm`, with local row `i` at global index
/// `rank_offset + i`. Each row is sent to its post office rank
/// (dolfinx::MPI::index_owner applied to the row's global index and
/// `shape[0]`), except rows for which the caller is itself the post
/// office, which are left in place.
///
/// @param[in] comm MPI communicator.
/// @param[in] x Local block of the array to distribute (row-major).
/// @param[in] shape Global shape of the array, `{num_rows, num_cols}`.
/// @param[in] rank_offset Global row index of local row 0 in `x`,
/// usually computed with `MPI_Exscan`.
/// @return (0) position of each received row within the caller's
/// post-office partition of `[0, shape[0])`, and (1) the received row
/// data (row-major). Rows for which the caller is itself the post
/// office are not included.
template <std::ranges::contiguous_range U>
std::pair<std::vector<std::int32_t>, std::vector<std::ranges::range_value_t<U>>>
distribute_to_postoffice(MPI_Comm comm, const U& x,
                         std::array<std::int64_t, 2> shape,
                         std::int64_t rank_offset);

/// @brief Fetch rows of a distributed row-major array via their post
/// office ranks.
///
/// `x` is a contiguous local block of a larger row-major array
/// distributed over `comm`, with local row `i` at global index
/// `rank_offset + i`. For each global row index in `indices` (which
/// may contain repeats), returns that row -- read directly from `x` if
/// already local, otherwise requested from its post office rank via
/// MPI neighbourhood collectives. Scalable provided each rank
/// exchanges with only a modest number of others.
///
/// @param[in] comm MPI communicator.
/// @param[in] indices Global row indices required by the caller.
/// @param[in] x Local block of the array to distribute (row-major).
/// @param[in] shape Global shape of the array, `{num_rows, num_cols}`.
/// @param[in] rank_offset Global row index of local row 0 in `x`
/// (usually an exclusive scan of row counts over the communicator `x`
/// is distributed across, which may differ from `comm`).
/// @return The row for each entry of `indices`, in the same order
/// (row-major storage).
/// @pre `shape[1] > 0`.
template <std::ranges::contiguous_range U>
std::vector<std::ranges::range_value_t<U>>
distribute_from_postoffice(MPI_Comm comm, std::span<const std::int64_t> indices,
                           const U& x, std::array<std::int64_t, 2> shape,
                           std::int64_t rank_offset);

/// @brief Distribute rows of a row-major array to the ranks that
/// require them, via the post office pattern.
///
/// Scalable provided each rank exchanges with only a modest number of
/// others.
///
/// @param[in] comm0 Communicator over which `indices` are resolved and
/// the result is returned.
/// @param[in] indices Global row indices required by the calling rank.
/// @param[in] comm1 Communicator across which `x` is distributed --
/// typically `comm0` itself or a sub-communicator of it, and
/// `MPI_COMM_NULL` on ranks where `x` is empty.
/// @param[in] x Local block of rows to distribute (row-major); local
/// row `i` has global index given by an exclusive scan of local row
/// counts over `comm1`.
/// @param[in] shape1 Number of columns of `x`.
/// @return The row for each entry of `indices`, in the same order
/// (row-major storage).
/// @pre `shape1 > 0`.
template <std::ranges::contiguous_range U>
std::vector<std::ranges::range_value_t<U>>
distribute_data(MPI_Comm comm0, std::span<const std::int64_t> indices,
                MPI_Comm comm1, const U& x, int shape1);

/// @private Type-dependent `false` for use in a `static_assert` that
/// should only fire when a template is actually instantiated (a bare
/// `static_assert(false, ...)` would fire unconditionally).
template <typename T>
struct dependent_false : std::false_type
{
};

/// @private Map a C++ scalar type to its MPI_Datatype. New types are
/// added here as another `else if constexpr` branch.
template <typename T>
MPI_Datatype mpi_datatype()
{
  if constexpr (std::same_as<T, float>)
    return MPI_FLOAT;
  else if constexpr (std::same_as<T, double>)
    return MPI_DOUBLE;
  else if constexpr (std::same_as<T, std::complex<float>>)
    return MPI_C_FLOAT_COMPLEX;
  else if constexpr (std::same_as<T, std::complex<double>>)
    return MPI_C_DOUBLE_COMPLEX;
  else if constexpr (std::same_as<T, std::int8_t>)
    return MPI_INT8_T;
  else if constexpr (std::same_as<T, std::int16_t>)
    return MPI_INT16_T;
  else if constexpr (std::same_as<T, std::int32_t>)
    return MPI_INT32_T;
  else if constexpr (std::same_as<T, std::int64_t>)
    return MPI_INT64_T;
  else if constexpr (std::same_as<T, std::uint8_t>)
    return MPI_UINT8_T;
  else if constexpr (std::same_as<T, std::uint16_t>)
    return MPI_UINT16_T;
  else if constexpr (std::same_as<T, std::uint32_t>)
    return MPI_UINT32_T;
  else if constexpr (std::same_as<T, std::uint64_t>)
    return MPI_UINT64_T;
  else
    static_assert(dependent_false<T>::value,
                  "No MPI datatype registered for this type.");
}

/// @brief Retrieves the MPI data type associated to the provided type.
/// @tparam T cpp type to map
template <typename T>
MPI_Datatype mpi_t = mpi_datatype<T>();

//---------------------------------------------------------------------------
namespace impl
{
/// @private Local (non-communicating) part of distribute_to_postoffice.
///
/// For each local row of `x`, determines its post office (destination)
/// rank by applying dolfinx::MPI::index_owner to the row's global
/// index and `shape0`, skipping rows already owned by this rank (which
/// never need to be sent anywhere). The remaining (destination,
/// position) pairs are grouped by destination with a radix sort --
/// this list can have hundreds of thousands of entries for a large
/// mesh/problem, too many for a generic comparison sort -- producing
/// the compact per-neighbour form (unique destination ranks, row
/// count per destination, and a local-row-to-neighbour-rank map)
/// that postoffice_exchange needs to drive MPI neighbourhood
/// collectives without a further per-row lookup during packing.
///
/// Factored out of distribute_to_postoffice so distribute_from_postoffice
/// can reuse the same planning step when it overlaps this push with
/// the NBX round that resolves who is requesting data from it.
///
/// @param[in] size Number of ranks in the communicator `x` is
/// distributed/required across.
/// @param[in] rank Rank of the caller in that communicator.
/// @param[in] shape0_local Number of local rows of `x` (before any
/// post-office redistribution).
/// @param[in] shape0 Global number of rows (row count summed over all
/// ranks).
/// @param[in] rank_offset Global row index of local row 0 in `x`.
/// @return (0) unique destination ranks, (1) number of rows destined
/// for each rank in (0), and (2) a map from local row index to
/// position in (0), or -1 if the row is already owned by this rank.
std::tuple<std::vector<int>, std::vector<std::int32_t>,
           std::vector<std::int32_t>>
postoffice_plan(int size, int rank, std::int32_t shape0_local,
                std::int64_t shape0, std::int64_t rank_offset);

/// @private Neighbourhood data exchange step of distribute_to_postoffice,
/// given an already-resolved `src` (e.g. from a single or overlapped
/// NBX consensus round).
/// @param[in] comm MPI communicator.
/// @param[in] x Local block of the array to distribute (row-major).
/// @param[in] shape Global shape of the array, `{num_rows, num_cols}`.
/// @param[in] rank_offset Global row index of local row 0 in `x`.
/// @param[in] dest Destination (post office) ranks, from
/// `postoffice_plan`.
/// @param[in] num_items_per_dest0 Number of rows destined for each
/// rank in `dest`, from `postoffice_plan`.
/// @param[in] pos_to_neigh_rank Map from local row index to position
/// in `dest`, from `postoffice_plan`.
/// @param[in] src Ranks that will send this rank data, i.e. the
/// incoming edges resolved from `dest` (e.g. via
/// `compute_graph_edges_nbx`).
/// @return (0) position of each received row within the caller's
/// post-office partition of `[0, shape[0])`, and (1) the received row
/// data (row-major).
template <std::ranges::contiguous_range U>
std::pair<std::vector<std::int32_t>, std::vector<std::ranges::range_value_t<U>>>
postoffice_exchange(MPI_Comm comm, const U& x,
                    std::array<std::int64_t, 2> shape, std::int64_t rank_offset,
                    std::span<const int> dest,
                    std::span<const std::int32_t> num_items_per_dest0,
                    std::span<const std::int32_t> pos_to_neigh_rank,
                    std::span<const int> src)
{
  using T = std::ranges::range_value_t<U>;

  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  assert(x.size() % shape[1] == 0);
  const std::int32_t shape0_local = x.size() / shape[1];

  // Create neighbourhood communicator for sending data to post offices
  MPI_Comm neigh_comm;
  int err = MPI_Dist_graph_create_adjacent(
      comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(), dest.data(),
      MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neigh_comm);
  dolfinx::MPI::check_error(comm, err);

  // Compute send displacements
  std::vector<std::int32_t> num_items_per_dest(num_items_per_dest0.begin(),
                                               num_items_per_dest0.end());
  std::vector<std::int32_t> send_disp{0};
  std::partial_sum(num_items_per_dest.begin(), num_items_per_dest.end(),
                   std::back_inserter(send_disp));

  // Pack send buffers
  std::vector<T> send_buffer_data(shape[1] * send_disp.back());
  std::vector<std::int64_t> send_buffer_index(send_disp.back());
  {
    std::vector<std::int32_t> send_offsets = send_disp;
    for (std::int32_t i = 0; i < shape0_local; ++i)
    {
      if (int neigh_dest = pos_to_neigh_rank[i]; neigh_dest != -1)
      {
        std::size_t pos = send_offsets[neigh_dest];
        send_buffer_index[pos] = i + rank_offset;
        std::copy_n(std::next(x.begin(), i * shape[1]), shape[1],
                    std::next(send_buffer_data.begin(), shape[1] * pos));
        ++send_offsets[neigh_dest];
      }
    }
  }

  // Send number of items to post offices (destination) that I will be
  // sending
  std::vector<int> num_items_recv(src.size());
  num_items_per_dest.reserve(1);
  num_items_recv.reserve(1);
  err = MPI_Neighbor_alltoall(num_items_per_dest.data(), 1, MPI_INT,
                              num_items_recv.data(), 1, MPI_INT, neigh_comm);
  dolfinx::MPI::check_error(comm, err);

  // Prepare receive displacement and buffers
  std::vector<std::int32_t> recv_disp(num_items_recv.size() + 1, 0);
  std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                   std::next(recv_disp.begin()));

  // Send/receive global indices
  std::vector<std::int64_t> recv_buffer_index(recv_disp.back());
  err = MPI_Neighbor_alltoallv(
      send_buffer_index.data(), num_items_per_dest.data(), send_disp.data(),
      MPI_INT64_T, recv_buffer_index.data(), num_items_recv.data(),
      recv_disp.data(), MPI_INT64_T, neigh_comm);
  dolfinx::MPI::check_error(comm, err);

  // Send/receive data (x)
  MPI_Datatype compound_type;
  MPI_Type_contiguous(shape[1], dolfinx::MPI::mpi_t<T>, &compound_type);
  MPI_Type_commit(&compound_type);
  std::vector<T> recv_buffer_data(shape[1] * recv_disp.back());
  err = MPI_Neighbor_alltoallv(
      send_buffer_data.data(), num_items_per_dest.data(), send_disp.data(),
      compound_type, recv_buffer_data.data(), num_items_recv.data(),
      recv_disp.data(), compound_type, neigh_comm);
  dolfinx::MPI::check_error(comm, err);
  err = MPI_Type_free(&compound_type);
  dolfinx::MPI::check_error(comm, err);
  err = MPI_Comm_free(&neigh_comm);
  dolfinx::MPI::check_error(comm, err);

  // Convert to local indices
  const std::int64_t r0 = common::local_range(rank, shape[0], size)[0];
  std::vector<std::int32_t> index_local(recv_buffer_index.size());
  std::ranges::transform(recv_buffer_index, index_local.begin(),
                         [r0](std::int64_t idx) { return idx - r0; });

  return {index_local, recv_buffer_data};
}
} // namespace impl

template <std::ranges::contiguous_range U>
std::pair<std::vector<std::int32_t>, std::vector<std::ranges::range_value_t<U>>>
distribute_to_postoffice(MPI_Comm comm, const U& x,
                         std::array<std::int64_t, 2> shape,
                         std::int64_t rank_offset)
{
  assert(rank_offset >= 0 or x.empty());
  assert(x.size() % shape[1] == 0);
  const std::int32_t shape0_local = x.size() / shape[1];

  spdlog::debug("Sending data to post offices (distribute_to_postoffice)");

  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  auto [dest, num_items_per_dest, pos_to_neigh_rank]
      = impl::postoffice_plan(size, rank, shape0_local, shape[0], rank_offset);

  // Determine source ranks
  const std::vector<int> src = MPI::compute_graph_edges_nbx(comm, dest);
  spdlog::info(
      "Number of neighbourhood source ranks in distribute_to_postoffice: {}",
      src.size());

  auto result
      = impl::postoffice_exchange(comm, x, shape, rank_offset, dest,
                                  num_items_per_dest, pos_to_neigh_rank, src);
  spdlog::debug("Completed send data to post offices.");
  return result;
}
//---------------------------------------------------------------------------
template <std::ranges::contiguous_range U>
std::vector<std::ranges::range_value_t<U>>
distribute_from_postoffice(MPI_Comm comm, std::span<const std::int64_t> indices,
                           const U& x, std::array<std::int64_t, 2> shape,
                           std::int64_t rank_offset)
{
  assert(rank_offset >= 0 or x.empty());
  using T = std::ranges::range_value_t<U>;

  common::Timer timer("Distribute row-wise data (scalable)");
  assert(shape[1] > 0);

  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  assert(x.size() % shape[1] == 0);
  const std::int64_t shape0_local = x.size() / shape[1];

  // 0. Send x data to/from post offices, and 1. determine which post
  //    office ranks hold the data I need (indices) -- these are
  //    independent local computations, so the two NBX consensus
  //    rounds they each need (below) are run concurrently in a single
  //    overlapped round rather than back-to-back.

  auto [send_dest, num_items_per_send_dest, pos_to_neigh_rank]
      = impl::postoffice_plan(size, rank, shape0_local, shape[0], rank_offset);

  // Build (src, global index, position) for each entry in 'indices'
  // not held locally, then sort. Locally-held entries are read
  // directly below -- skipping them here avoids a wasted round trip.
  std::vector<std::tuple<int, std::int64_t, std::int32_t>> src_to_index;
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    std::int64_t idx = indices[i];
    if (idx >= rank_offset and idx < rank_offset + shape0_local)
      continue;
    if (int src = dolfinx::MPI::index_owner(size, idx, shape[0]); src != rank)
      src_to_index.push_back({src, idx, i});
  }

  // Radix sort on the rank alone (not the full tuple) -- only
  // grouping by rank matters below; order within a group doesn't.
  {
    std::vector<std::int32_t> perm(src_to_index.size());
    std::iota(perm.begin(), perm.end(), 0);
    dolfinx::radix_sort(perm, [&src_to_index](std::int32_t i)
                        { return std::get<0>(src_to_index[i]); });
    std::vector<std::tuple<int, std::int64_t, std::int32_t>> sorted(
        src_to_index.size());
    for (std::size_t i = 0; i < perm.size(); ++i)
      sorted[i] = src_to_index[perm[i]];
    src_to_index = std::move(sorted);
  }

  // Build list of neighbour src ranks and count number of items (rows
  // of x) to receive from each src post office (by neighbourhood rank)
  std::vector<std::int32_t> num_items_per_src;
  std::vector<int> src;
  {
    auto it = src_to_index.begin();
    while (it != src_to_index.end())
    {
      src.push_back(std::get<0>(*it));
      auto it1
          = std::find_if(it, src_to_index.end(), [r = src.back()](auto& idx)
                         { return std::get<0>(idx) != r; });
      num_items_per_src.push_back(std::distance(it, it1));
      it = it1;
    }
  }

  // Determine, in one overlapped NBX round, (0) the post office ranks
  // that hold data for me to receive (post office send round) and (1)
  // the 'delivery' destination ranks that want data from me (my
  // request round)
  auto [post_src, dest] = dolfinx::MPI::compute_graph_edges_nbx(
      comm, send_dest, static_cast<int>(tag::consensus_nbx), src,
      static_cast<int>(tag::consensus_nbx) + 1);
  spdlog::info(
      "Neighbourhood destination ranks from post office in "
      "distribute_data (rank, num dests, num dests/mpi_size): {}, {}, {}",
      rank, dest.size(), static_cast<double>(dest.size()) / size);

  // Send receive x data to post office (only for rows that need to be
  // communicated)
  auto [post_indices, post_x] = impl::postoffice_exchange(
      comm, x, {shape[0], shape[1]}, rank_offset, send_dest,
      num_items_per_send_dest, pos_to_neigh_rank, post_src);
  assert(post_indices.size() == post_x.size() / shape[1]);

  // Create neighbourhood communicator for sending data to post offices
  // (src), and receiving data form my send my post office
  MPI_Comm neigh_comm0;
  int err = MPI_Dist_graph_create_adjacent(
      comm, dest.size(), dest.data(), MPI_UNWEIGHTED, src.size(), src.data(),
      MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neigh_comm0);
  dolfinx::MPI::check_error(comm, err);

  // Communicate number of requests to each source
  std::vector<int> num_items_recv(dest.size());
  num_items_per_src.reserve(1);
  num_items_recv.reserve(1);
  err = MPI_Neighbor_alltoall(num_items_per_src.data(), 1, MPI_INT,
                              num_items_recv.data(), 1, MPI_INT, neigh_comm0);
  dolfinx::MPI::check_error(comm, err);

  // Prepare send/receive displacements
  std::vector<std::int32_t> send_disp{0};
  std::partial_sum(num_items_per_src.begin(), num_items_per_src.end(),
                   std::back_inserter(send_disp));
  std::vector<std::int32_t> recv_disp = {0};
  std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                   std::back_inserter(recv_disp));

  // Pack my requested indices (global) in send buffer ready to send to
  // post offices
  assert(send_disp.back() == static_cast<int>(src_to_index.size()));
  std::vector<std::int64_t> send_buffer_index(src_to_index.size());
  std::ranges::transform(src_to_index, send_buffer_index.begin(),
                         [](auto x) { return std::get<1>(x); });

  // Prepare the receive buffer
  std::vector<std::int64_t> recv_buffer_index(recv_disp.back());
  err = MPI_Neighbor_alltoallv(
      send_buffer_index.data(), num_items_per_src.data(), send_disp.data(),
      MPI_INT64_T, recv_buffer_index.data(), num_items_recv.data(),
      recv_disp.data(), MPI_INT64_T, neigh_comm0);
  dolfinx::MPI::check_error(comm, err);

  err = MPI_Comm_free(&neigh_comm0);
  dolfinx::MPI::check_error(comm, err);

  // 2. Send data (rows of x) from post office back to requesting ranks
  //    (transpose of the preceding communication pattern operation)

  // Build map from local index to post_indices position. Set to -1 for
  // data that was already on this rank and was therefore was not
  // sent/received via a postoffice.
  const std::array<std::int64_t, 2> postoffice_range
      = common::local_range(rank, shape[0], size);
  std::vector<std::int32_t> post_indices_map(
      postoffice_range[1] - postoffice_range[0], -1);
  for (std::size_t i = 0; i < post_indices.size(); ++i)
  {
    assert(post_indices[i] < static_cast<int>(post_indices_map.size()));
    post_indices_map[post_indices[i]] = i;
  }

  // Build send buffer
  std::vector<T> send_buffer_data(shape[1] * recv_disp.back());
  for (std::int32_t i = 0; i < recv_disp.back(); ++i)
  {
    std::int64_t index = recv_buffer_index[i];
    if (index >= rank_offset and index < (rank_offset + shape0_local))
    {
      // I already had this index before any communication
      std::int32_t local_index = index - rank_offset;
      std::copy_n(std::next(x.begin(), shape[1] * local_index), shape[1],
                  std::next(send_buffer_data.begin(), shape[1] * i));
    }
    else
    {
      // Take from my 'post bag'
      std::int64_t local_index = index - postoffice_range[0];
      std::int32_t pos = post_indices_map[local_index];
      assert(pos != -1);
      std::copy_n(std::next(post_x.begin(), shape[1] * pos), shape[1],
                  std::next(send_buffer_data.begin(), shape[1] * i));
    }
  }

  err = MPI_Dist_graph_create_adjacent(
      comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(), dest.data(),
      MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neigh_comm0);
  dolfinx::MPI::check_error(comm, err);

  MPI_Datatype compound_type0;
  MPI_Type_contiguous(shape[1], dolfinx::MPI::mpi_t<T>, &compound_type0);
  MPI_Type_commit(&compound_type0);

  std::vector<T> recv_buffer_data(shape[1] * send_disp.back());
  err = MPI_Neighbor_alltoallv(
      send_buffer_data.data(), num_items_recv.data(), recv_disp.data(),
      compound_type0, recv_buffer_data.data(), num_items_per_src.data(),
      send_disp.data(), compound_type0, neigh_comm0);
  dolfinx::MPI::check_error(comm, err);

  err = MPI_Type_free(&compound_type0);
  dolfinx::MPI::check_error(comm, err);
  err = MPI_Comm_free(&neigh_comm0);
  dolfinx::MPI::check_error(comm, err);

  std::vector<std::int32_t> index_pos_to_buffer(indices.size(), -1);
  for (std::size_t i = 0; i < src_to_index.size(); ++i)
    index_pos_to_buffer[std::get<2>(src_to_index[i])] = i;

  // Extra data to return
  std::vector<T> x_new(shape[1] * indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    const std::int64_t index = indices[i];
    if (index >= rank_offset and index < (rank_offset + shape0_local))
    {
      // Had data from the start in x
      std::int64_t local_index = index - rank_offset;
      std::copy_n(std::next(x.begin(), shape[1] * local_index), shape[1],
                  std::next(x_new.begin(), shape[1] * i));
    }
    else if (std::int32_t pos = index_pos_to_buffer[i]; pos != -1)
    {
      // In my received post: index_pos_to_buffer[i] != -1 iff
      // index_owner would say this rank isn't the owner -- avoids
      // recomputing it.
      std::copy_n(std::next(recv_buffer_data.begin(), shape[1] * pos), shape[1],
                  std::next(x_new.begin(), shape[1] * i));
    }
    else
    {
      // In my post office bag
      std::int64_t local_index = index - postoffice_range[0];
      std::int32_t bag_pos = post_indices_map[local_index];
      assert(bag_pos != -1);
      std::copy_n(std::next(post_x.begin(), shape[1] * bag_pos), shape[1],
                  std::next(x_new.begin(), shape[1] * i));
    }
  }

  return x_new;
}
//---------------------------------------------------------------------------
template <std::ranges::contiguous_range U>
std::vector<std::ranges::range_value_t<U>>
distribute_data(MPI_Comm comm0, std::span<const std::int64_t> indices,
                MPI_Comm comm1, const U& x, int shape1)
{
  assert(shape1 > 0);
  assert(x.size() % shape1 == 0);
  const std::int64_t shape0_local = x.size() / shape1;

  std::int64_t shape0 = 0;
  int err
      = MPI_Allreduce(&shape0_local, &shape0, 1, MPI_INT64_T, MPI_SUM, comm0);
  dolfinx::MPI::check_error(comm0, err);

  std::int64_t rank_offset = -1;
  if (comm1 != MPI_COMM_NULL)
  {
    rank_offset = 0;
    err = MPI_Exscan(&shape0_local, &rank_offset, 1,
                     dolfinx::MPI::mpi_t<std::int64_t>, MPI_SUM, comm1);
    dolfinx::MPI::check_error(comm1, err);
  }
  else if (!x.empty())
    throw std::runtime_error("Non-empty data on null MPI communicator");

  return distribute_from_postoffice(comm0, indices, x, {shape0, shape1},
                                    rank_offset);
}
//---------------------------------------------------------------------------

} // namespace dolfinx::MPI
