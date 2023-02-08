// Copyright (C) 2007-2014 Magnus Vikstrøm and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <numeric>
#include <set>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#define MPICH_IGNORE_CXX_SEEK 1
#include <mpi.h>

/// MPI support functionality
namespace dolfinx::MPI
{

/// MPI communication tags
enum class tag : int
{
  consensus_pcx,
  consensus_pex
};

/// A duplicate MPI communicator and manage lifetime of the
/// communicator
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
/// communicator
int size(MPI_Comm comm);

/// @brief Check MPI error code. If the error code is not equal to
/// MPI_SUCCESS, then std::abort is called.
/// @param[in] comm MPI communicator
/// @param[in] code Error code returned by an MPI function call
void check_error(MPI_Comm comm, int code);

/// @brief Return local range for the calling process, partitioning the
/// global [0, N - 1] range across all ranks into partitions of almost
/// equal size.
/// @param[in] rank MPI rank of the caller
/// @param[in] N The value to partition
/// @param[in] size The number of MPI ranks across which to partition
/// `N`
constexpr std::array<std::int64_t, 2> local_range(int rank, std::int64_t N,
                                                  int size)
{
  assert(rank >= 0);
  assert(N >= 0);
  assert(size > 0);

  // Compute number of items per rank and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  if (rank < r)
    return {rank * (n + 1), rank * (n + 1) + n + 1};
  else
    return {rank * n + r, rank * n + r + n};
}

/// @brief Return which rank owns index in global range [0, N - 1]
/// (inverse of MPI::local_range).
/// @param[in] size Number of MPI ranks
/// @param[in] index The index to determine owning rank
/// @param[in] N Total number of indices
/// @return The rank of the owning process
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
/// @note For sparse graphs, this function has \f$O(\log p)\f$ cost,
/// where \f$p\f$is the number of MPI ranks. It is suitable for modest
/// MPI rank counts.
///
/// @note The order of the returned ranks is not deterministic.
///
/// @note Collective.
///
/// @param[in] comm MPI communicator
/// @param[in] edges Edges (ranks) from this rank (the caller).
/// @return Ranks that have defined edges from them to this rank.
std::vector<int> compute_graph_edges_nbx(MPI_Comm comm,
                                         std::span<const int> edges);

/// @brief Distribute row data to 'post office' ranks.
///
/// This function takes row-wise data that is distributed across
/// processes. Data is not duplicated across ranks. The global index of
/// a row is its local row position plus the offset for the calling
/// process. The post office rank for a row is determined by applying
/// MPI::index_owner to the global index, and the row is then sent to
/// the post office rank. The function returns that row data for which
/// the caller is the post office.
///
/// @param[in] comm MPI communicator
/// @param[in] x Data to distribute (2D, row-major layout)
/// @param[in] shape The global shape of `x`
/// @param[in] rank_offset The rank offset such that global index of
/// local row `i` in `x` is `rank_offset + i`. It is usually computed
/// using `MPI_Exscan`.
/// @returns (0) local indices of my post office data and (1) the data
/// (row-major). It **does not** include rows that are in `x`, i.e. rows
/// for which the calling process is the post office
template <typename T>
std::pair<std::vector<std::int32_t>, std::vector<T>>
distribute_to_postoffice(MPI_Comm comm, std::span<const T> x,
                         std::array<std::int64_t, 2> shape,
                         std::int64_t rank_offset);

/// @brief Distribute rows of a rectangular data array from post office
/// ranks to ranks where they are required.
///
/// This function determines local neighborhoods for communication, and
/// then using MPI neighbourhood collectives to exchange data. It is
/// scalable if the neighborhoods are relatively small, i.e. each
/// process communicated with a modest number of other processes
///
/// @param[in] comm The MPI communicator
/// @param[in] indices Global indices of the data (row indices) required
/// by calling process
/// @param[in] x Data (2D array, row-major) on calling process which may
/// be distributed (by row). The global index for the `[0, ..., n)`
/// local rows is assumed to be the local index plus the offset for this
/// rank.
/// @param[in] shape The global shape of `x`
/// @param[in] rank_offset The rank offset such that global index of
/// local row `i` in `x` is `rank_offset + i`. It is usually computed
/// using `MPI_Exscan`.
/// @return The data for each index in `indices` (row-major storage)
/// @pre `shape1 > 0`
template <typename T>
std::vector<T> distribute_from_postoffice(MPI_Comm comm,
                                          std::span<const std::int64_t> indices,
                                          std::span<const T> x,
                                          std::array<std::int64_t, 2> shape,
                                          std::int64_t rank_offset);

/// @brief Distribute rows of a rectangular data array to ranks where
/// they are required (scalable version).
///
/// This function determines local neighborhoods for communication, and
/// then using MPI neighbourhood collectives to exchange data. It is
/// scalable if the neighborhoods are relatively small, i.e. each
/// process communicated with a modest number of other processes.
///
/// @note The non-scalable version of this function,
/// MPI::distribute_data1, can be faster up to some number of MPI ranks
/// with number of ranks depending on the locality of the data, the MPI
/// implementation and the network.
///
/// @param[in] comm The MPI communicator
/// @param[in] indices Global indices of the data (row indices) required
/// by calling process
/// @param[in] x Data (2D array, row-major) on calling process which may
/// be distributed (by row). The global index for the `[0, ..., n)`
/// local rows is assumed to be the local index plus the offset for this
/// rank.
/// @param[in] shape1 The number of columns of the data array `x`.
/// @return The data for each index in `indices` (row-major storage)
/// @pre `shape1 > 0`
template <typename T>
std::vector<T> distribute_data(MPI_Comm comm,
                               std::span<const std::int64_t> indices,
                               std::span<const T> x, int shape1);

template <typename T>
struct dependent_false : std::false_type
{
};

/// MPI Type
template <typename T>
constexpr MPI_Datatype mpi_type()
{
  if constexpr (std::is_same_v<T, float>)
    return MPI_FLOAT;
  else if constexpr (std::is_same_v<T, double>)
    return MPI_DOUBLE;
  else if constexpr (std::is_same_v<T, std::complex<double>>)
    return MPI_C_DOUBLE_COMPLEX;
  else if constexpr (std::is_same_v<T, std::complex<float>>)
    return MPI_C_FLOAT_COMPLEX;
  else if constexpr (std::is_same_v<T, short int>)
    return MPI_SHORT;
  else if constexpr (std::is_same_v<T, int>)
    return MPI_INT;
  else if constexpr (std::is_same_v<T, unsigned int>)
    return MPI_UNSIGNED;
  else if constexpr (std::is_same_v<T, long int>)
    return MPI_LONG;
  else if constexpr (std::is_same_v<T, unsigned long>)
    return MPI_UNSIGNED_LONG;
  else if constexpr (std::is_same_v<T, long long>)
    return MPI_LONG_LONG;
  else if constexpr (std::is_same_v<T, unsigned long long>)
    return MPI_UNSIGNED_LONG_LONG;
  else if constexpr (std::is_same_v<T, bool>)
    return MPI_C_BOOL;
  else if constexpr (std::is_same_v<T, std::int8_t>)
    return MPI_INT8_T;
  else
    // Issue compile time error
    static_assert(!std::is_same_v<T, T>);
}

//---------------------------------------------------------------------------
template <typename T>
std::pair<std::vector<std::int32_t>, std::vector<T>>
distribute_to_postoffice(MPI_Comm comm, std::span<const T> x,
                         std::array<std::int64_t, 2> shape,
                         std::int64_t rank_offset)
{
  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  assert(x.size() % shape[1] == 0);
  const std::int32_t shape0_local = x.size() / shape[1];

  LOG(2) << "Sending data to post offices (distribute_to_postoffice)";

  // Post office ranks will receive data from this rank
  std::vector<int> row_to_dest(shape0_local);
  for (std::int32_t i = 0; i < shape0_local; ++i)
  {
    int dest = MPI::index_owner(size, i + rank_offset, shape[0]);
    row_to_dest[i] = dest;
  }

  // Build list of (dest, positions) for each row that doesn't belong to
  // this rank, then sort
  std::vector<std::array<std::int32_t, 2>> dest_to_index;
  dest_to_index.reserve(shape0_local);
  for (std::int32_t i = 0; i < shape0_local; ++i)
  {
    std::size_t idx = i + rank_offset;
    if (int dest = MPI::index_owner(size, idx, shape[0]); dest != rank)
      dest_to_index.push_back({dest, i});
  }
  std::sort(dest_to_index.begin(), dest_to_index.end());

  // Build list of neighbour src ranks and count number of items (rows
  // of x) to receive from each src post office (by neighbourhood rank)
  std::vector<int> dest;
  std::vector<std::int32_t> num_items_per_dest,
      pos_to_neigh_rank(shape0_local, -1);
  {
    auto it = dest_to_index.begin();
    while (it != dest_to_index.end())
    {
      const int neigh_rank = dest.size();

      // Store global rank
      dest.push_back((*it)[0]);

      // Find iterator to next global rank
      auto it1
          = std::find_if(it, dest_to_index.end(),
                         [r = dest.back()](auto& idx) { return idx[0] != r; });

      // Store number of items for current rank
      num_items_per_dest.push_back(std::distance(it, it1));

      // Map from local x index to local destination rank
      for (auto e = it; e != it1; ++e)
        pos_to_neigh_rank[(*e)[1]] = neigh_rank;

      // Advance iterator
      it = it1;
    }
  }

  // Determine source ranks
  const std::vector<int> src = MPI::compute_graph_edges_nbx(comm, dest);
  LOG(INFO)
      << "Number of neighbourhood source ranks in distribute_to_postoffice: "
      << src.size();

  // Create neighbourhood communicator for sending data to post offices
  MPI_Comm neigh_comm;
  int err = MPI_Dist_graph_create_adjacent(
      comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(), dest.data(),
      MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neigh_comm);
  dolfinx::MPI::check_error(comm, err);

  // Compute send displacements
  std::vector<std::int32_t> send_disp = {0};
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
  MPI_Type_contiguous(shape[1], dolfinx::MPI::mpi_type<T>(), &compound_type);
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

  LOG(2) << "Completed send data to post offices.";

  // Convert to local indices
  const std::int64_t r0 = MPI::local_range(rank, shape[0], size)[0];
  std::vector<std::int32_t> index_local(recv_buffer_index.size());
  std::transform(recv_buffer_index.cbegin(), recv_buffer_index.cend(),
                 index_local.begin(), [r0](auto idx) { return idx - r0; });

  return {index_local, recv_buffer_data};
}
//---------------------------------------------------------------------------
template <typename T>
std::vector<T> distribute_from_postoffice(MPI_Comm comm,
                                          std::span<const std::int64_t> indices,
                                          std::span<const T> x,
                                          std::array<std::int64_t, 2> shape,
                                          std::int64_t rank_offset)
{
  common::Timer timer("Distribute row-wise data (scalable)");
  assert(shape[1] > 0);

  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  assert(x.size() % shape[1] == 0);
  const std::int64_t shape0_local = x.size() / shape[1];

  // 0. Send x data to/from post offices

  // Send receive x data to post office (only for rows that need to be
  // communicated)
  auto [post_indices, post_x] = MPI::distribute_to_postoffice(
      comm, x, {shape[0], shape[1]}, rank_offset);
  assert(post_indices.size() == post_x.size() / shape[1]);

  // 1. Send request to post office ranks for data

  // Build list of (src, global index, global, index positions) for each
  // entry in 'indices' that doesn't belong to this rank, then sort
  std::vector<std::tuple<int, std::int64_t, std::int32_t>> src_to_index;
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    std::size_t idx = indices[i];
    if (int src = MPI::index_owner(size, idx, shape[0]); src != rank)
      src_to_index.push_back({src, idx, i});
  }
  std::sort(src_to_index.begin(), src_to_index.end());

  // Build list is neighbour src ranks and count number of items (rows
  // of x) to receive from each src post office (by neighbourhood rank)
  std::vector<std::int32_t> num_items_per_src;
  std::vector<int> src;
  {
    auto it = src_to_index.begin();
    while (it != src_to_index.end())
    {
      src.push_back(std::get<0>(*it));
      auto it1 = std::find_if(it, src_to_index.end(),
                              [r = src.back()](auto& idx)
                              { return std::get<0>(idx) != r; });
      num_items_per_src.push_back(std::distance(it, it1));
      it = it1;
    }
  }

  // Determine 'delivery' destination ranks (ranks that want data from
  // me)
  const std::vector<int> dest
      = dolfinx::MPI::compute_graph_edges_nbx(comm, src);
  LOG(INFO) << "Neighbourhood destination ranks from post office in "
               "distribute_data (rank, num dests, num dests/mpi_size): "
            << rank << ", " << dest.size() << ", "
            << static_cast<double>(dest.size()) / size;

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
  std::vector<std::int32_t> send_disp = {0};
  std::partial_sum(num_items_per_src.begin(), num_items_per_src.end(),
                   std::back_inserter(send_disp));
  std::vector<std::int32_t> recv_disp = {0};
  std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                   std::back_inserter(recv_disp));

  // Pack my requested indices (global) in send buffer ready to send to
  // post offices
  assert(send_disp.back() == (int)src_to_index.size());
  std::vector<std::int64_t> send_buffer_index(src_to_index.size());
  std::transform(src_to_index.cbegin(), src_to_index.cend(),
                 send_buffer_index.begin(),
                 [](auto& x) { return std::get<1>(x); });

  // Prepare the receive buffer
  std::vector<std::int64_t> recv_buffer_index(recv_disp.back());
  err = MPI_Neighbor_alltoallv(
      send_buffer_index.data(), num_items_per_src.data(), send_disp.data(),
      MPI_INT64_T, recv_buffer_index.data(), num_items_recv.data(),
      recv_disp.data(), MPI_INT64_T, neigh_comm0);
  dolfinx::MPI::check_error(comm, err);

  err = MPI_Comm_free(&neigh_comm0);
  dolfinx::MPI::check_error(comm, err);

  // 2. Send data (rows of x) back to requesting ranks (transpose of the
  // preceding communication pattern operation)

  // Build map from local index to post_indices position. Set to -1 for
  // data that was already on this rank and was therefore was not
  // sent/received via a postoffice.
  const std::array<std::int64_t, 2> postoffice_range
      = MPI::local_range(rank, shape[0], size);
  std::vector<std::int32_t> post_indices_map(
      postoffice_range[1] - postoffice_range[0], -1);
  for (std::size_t i = 0; i < post_indices.size(); ++i)
  {
    assert(post_indices[i] < (int)post_indices_map.size());
    post_indices_map[post_indices[i]] = i;
  }

  // Build send buffer
  std::vector<T> send_buffer_data(shape[1] * recv_disp.back());
  for (std::size_t p = 0; p < recv_disp.size() - 1; ++p)
  {
    int offset = recv_disp[p];
    for (std::int32_t i = recv_disp[p]; i < recv_disp[p + 1]; ++i)
    {
      std::int64_t index = recv_buffer_index[i];
      if (index >= rank_offset and index < (rank_offset + shape0_local))
      {
        // I already had this index before any communication
        std::int32_t local_index = index - rank_offset;
        std::copy_n(std::next(x.begin(), shape[1] * local_index), shape[1],
                    std::next(send_buffer_data.begin(), shape[1] * offset));
      }
      else
      {
        // Take from my 'post bag'
        auto local_index = index - postoffice_range[0];
        std::int32_t pos = post_indices_map[local_index];
        assert(pos != -1);
        std::copy_n(std::next(post_x.begin(), shape[1] * pos), shape[1],
                    std::next(send_buffer_data.begin(), shape[1] * offset));
      }

      ++offset;
    }
  }

  err = MPI_Dist_graph_create_adjacent(
      comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(), dest.data(),
      MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neigh_comm0);
  dolfinx::MPI::check_error(comm, err);

  MPI_Datatype compound_type0;
  MPI_Type_contiguous(shape[1], dolfinx::MPI::mpi_type<T>(), &compound_type0);
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
      auto local_index = index - rank_offset;
      std::copy_n(std::next(x.begin(), shape[1] * local_index), shape[1],
                  std::next(x_new.begin(), shape[1] * i));
    }
    else
    {
      if (int src = MPI::index_owner(size, index, shape[0]); src == rank)
      {
        // In my post office bag
        auto local_index = index - postoffice_range[0];
        std::int32_t pos = post_indices_map[local_index];
        assert(pos != -1);
        std::copy_n(std::next(post_x.begin(), shape[1] * pos), shape[1],
                    std::next(x_new.begin(), shape[1] * i));
      }
      else
      {
        // In my received post
        std::int32_t pos = index_pos_to_buffer[i];
        assert(pos != -1);
        std::copy_n(std::next(recv_buffer_data.begin(), shape[1] * pos),
                    shape[1], std::next(x_new.begin(), shape[1] * i));
      }
    }
  }

  return x_new;
}
//---------------------------------------------------------------------------
template <typename T>
std::vector<T> distribute_data(MPI_Comm comm,
                               std::span<const std::int64_t> indices,
                               std::span<const T> x, int shape1)
{
  assert(shape1 > 0);
  assert(x.size() % shape1 == 0);
  const std::int64_t shape0_local = x.size() / shape1;

  std::int64_t shape0(0), rank_offset(0);
  int err
      = MPI_Allreduce(&shape0_local, &shape0, 1, MPI_INT64_T, MPI_SUM, comm);
  dolfinx::MPI::check_error(comm, err);
  err = MPI_Exscan(&shape0_local, &rank_offset, 1, MPI_INT64_T, MPI_SUM, comm);
  dolfinx::MPI::check_error(comm, err);

  return distribute_from_postoffice(comm, indices, x, {shape0, shape1},
                                    rank_offset);
}
//---------------------------------------------------------------------------

} // namespace dolfinx::MPI
