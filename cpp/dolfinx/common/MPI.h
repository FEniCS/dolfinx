// Copyright (C) 2007-2014 Magnus Vikstrøm and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <numeric>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

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

/// @brief Return local range for given process, splitting [0, N - 1]
/// into size() portions of almost equal size.
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

/// @brief Return which rank owns index (inverse of MPI::local_range)
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

/// @brief Return list of neighbors (sources and destinations) for a
/// neighborhood communicator.
/// @param[in] comm Neighborhood communicator
/// @return source ranks [0], destination ranks [1]
std::array<std::vector<int>, 2> neighbors(MPI_Comm comm);

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
/// @note Collective
///
/// @param[in] comm MPI communicator
/// @param[in] edges Edges (ranks) from this rank (the caller).
/// @return Ranks that have defined edges from them to this rank.
std::vector<int> compute_graph_edges_pcx(MPI_Comm comm,
                                         const xtl::span<const int>& edges);

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
/// @note Collective over ranks that are connected by graph edge.
///
/// @param[in] comm MPI communicator
/// @param[in] edges Edges (ranks) from this rank (the caller).
/// @return Ranks that have defined edges from them to this rank.
std::vector<int> compute_graph_edges_nbx(MPI_Comm comm,
                                         const xtl::span<const int>& edges);

/// @brief Distribute row data to 'post office' rank.
///
/// The post office rank for a row is determined by applying
/// MPI::index_owner global index of the row.
///
/// @param[in] comm MPI communicator
/// @param[in] x Data to distribute (2D, row-major layout)
/// @param[in] shape The global shape of `x`
/// @param[in] rank_offset The rank offset such that global index of
/// local row `i` in `x` is `rank_offset + i`. It is usually computed
/// using `MPI_Exscan`.
/// @returns (0) global indices of my post office data and (1) the data
/// (row-major). It **does not** include rows that are in `x`, i.e. rows
/// for which the calling process is the post office
template <typename T>
std::pair<std::vector<std::int32_t>, std::vector<T>>
distribute_to_postoffice(MPI_Comm comm, const xtl::span<const T>& x,
                         std::array<std::int64_t, 2> shape,
                         std::int64_t rank_offset);

/// @brief Distribute rows of a rectangular data array from post office
/// ranks to ranks where they are required.
///
/// This functions determines local neighborhoods for communication, and
/// then using MPI neighbourhood collectives to exchange data. It is
/// scalable if the neighborhoods are relatively small, i.e. each process
/// communicated with a modest number of othe processes/
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
/// @param[in] shape The global shape of `x`
/// @param[in] rank_offset The rank offset such that global index of
/// local row `i` in `x` is `rank_offset + i`. It is usually computed
/// using `MPI_Exscan`.
/// @return The data for each index in `indices` (row-major storage)
/// @pre `shape1 > 0`
template <typename T>
std::vector<T> distribute_from_postoffice(
    MPI_Comm comm, const xtl::span<const std::int64_t>& indices,
    const xtl::span<const T>& x, std::array<std::int64_t, 2> shape,
    std::int64_t rank_offset);

/// @brief Distribute rows of a rectangular data array to ranks where
/// they are required.
///
/// This functions determines local neighborhoods for communication, and
/// then using MPI neighbourhood collectives to exchange data. It is
/// scalable if the neighborhoods are relatively small, i.e. each process
/// communicated with a modest number of othe processes/
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
                               const xtl::span<const std::int64_t>& indices,
                               const xtl::span<const T>& x, int shape1);

/// @brief Distribute rows of a rectangular data array to ranks where
/// they are required.
///
/// See MPI::distribute_data description.
///
/// @note This functions used MPI all-to-all collectives and is
/// therefore not scalable. It is typically faster the the scalable
/// MPI::distribute_data function up to some number of MPI ranks.
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
std::vector<T> distribute_data1(MPI_Comm comm,
                                const xtl::span<const std::int64_t>& indices,
                                const xtl::span<const T>& x, int shape1);

/// @todo Remove this function
/// Send in_values[p0] to process p0 and receive values from process
/// p1 in out_values[p1]
template <typename T>
graph::AdjacencyList<T> all_to_all(MPI_Comm comm,
                                   const graph::AdjacencyList<T>& send_data);

/// @brief Send in_values[n0] to neighbor process n0 and receive values
/// from neighbor process n1 in out_values[n1].
/// @param[in] comm Neighborhood communicator
/// @param[in] send_data The data to send to each rank.
/// graph::AdjacencyList<T>::num_nodes should be equal to the number of
/// neighbourhood out edges.
/// @return Data received from incoming neighbourhood ranks.
template <typename T>
graph::AdjacencyList<T>
neighbor_all_to_all(MPI_Comm comm, const graph::AdjacencyList<T>& send_data);

template <typename T>
struct dependent_false : std::false_type
{
};

/// MPI Type
template <typename T>
constexpr MPI_Datatype mpi_type()
{
  if constexpr (std::is_same<T, float>::value)
    return MPI_FLOAT;
  else if constexpr (std::is_same<T, double>::value)
    return MPI_DOUBLE;
  else if constexpr (std::is_same<T, std::complex<double>>::value)
    return MPI_C_DOUBLE_COMPLEX;
  else if constexpr (std::is_same<T, std::complex<float>>::value)
    return MPI_C_FLOAT_COMPLEX;
  else if constexpr (std::is_same<T, short int>::value)
    return MPI_SHORT;
  else if constexpr (std::is_same<T, int>::value)
    return MPI_INT;
  else if constexpr (std::is_same<T, unsigned int>::value)
    return MPI_UNSIGNED;
  else if constexpr (std::is_same<T, long int>::value)
    return MPI_LONG;
  else if constexpr (std::is_same<T, unsigned long>::value)
    return MPI_UNSIGNED_LONG;
  else if constexpr (std::is_same<T, long long>::value)
    return MPI_LONG_LONG;
  else if constexpr (std::is_same<T, unsigned long long>::value)
    return MPI_UNSIGNED_LONG_LONG;
  else if constexpr (std::is_same<T, bool>::value)
    return MPI_C_BOOL;
  else if constexpr (std::is_same<T, std::int8_t>::value)
    return MPI_INT8_T;
  else
    // Issue compile time error
    static_assert(!std::is_same<T, T>::value);
}

//---------------------------------------------------------------------------
template <typename T>
std::pair<std::vector<std::int32_t>, std::vector<T>>
distribute_to_postoffice(MPI_Comm comm, const xtl::span<const T>& x,
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

  // Build list is neighbour src ranks and count number of items (rows
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
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm);

  // Compute send displacements
  std::vector<std::int32_t> send_disp = {0};
  std::partial_sum(num_items_per_dest.begin(), num_items_per_dest.end(),
                   std::back_insert_iterator(send_disp));

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
  MPI_Neighbor_alltoall(num_items_per_dest.data(), 1, MPI_INT,
                        num_items_recv.data(), 1, MPI_INT, neigh_comm);

  // Prepare receive displacement and buffers
  std::vector<std::int32_t> recv_disp = {0};
  std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                   std::back_insert_iterator(recv_disp));

  // Send/receive global indices
  std::vector<std::int64_t> recv_buffer_index(recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer_index.data(), num_items_per_dest.data(),
                         send_disp.data(), MPI_INT64_T,
                         recv_buffer_index.data(), num_items_recv.data(),
                         recv_disp.data(), MPI_INT64_T, neigh_comm);

  // Send/receive data (x)
  MPI_Datatype compound_type;
  MPI_Type_contiguous(shape[1], dolfinx::MPI::mpi_type<T>(), &compound_type);
  MPI_Type_commit(&compound_type);
  std::vector<T> recv_buffer_data(shape[1] * recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer_data.data(), num_items_per_dest.data(),
                         send_disp.data(), compound_type,
                         recv_buffer_data.data(), num_items_recv.data(),
                         recv_disp.data(), compound_type, neigh_comm);
  MPI_Type_free(&compound_type);

  MPI_Comm_free(&neigh_comm);

  LOG(2) << "Completed send data to post offices.";

  // Convert to local indices
  const std::int64_t r0 = MPI::local_range(rank, shape[0], size)[0];
  std::vector<std::int32_t> index_local(recv_buffer_index.size());
  std::transform(recv_buffer_index.cbegin(), recv_buffer_index.cend(),
                 index_local.begin(), [r0](auto idx) { return idx - r0; });

  return {index_local, recv_buffer_data};
};
//---------------------------------------------------------------------------
template <typename T>
std::vector<T> distribute_from_postoffice(
    MPI_Comm comm, const xtl::span<const std::int64_t>& indices,
    const xtl::span<const T>& x, std::array<std::int64_t, 2> shape,
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
  MPI_Dist_graph_create_adjacent(comm, dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 src.size(), src.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm0);

  // Communicate number of requests to each source
  std::vector<int> num_items_recv(dest.size());
  num_items_per_src.reserve(1);
  num_items_recv.reserve(1);
  MPI_Neighbor_alltoall(num_items_per_src.data(), 1, MPI_INT,
                        num_items_recv.data(), 1, MPI_INT, neigh_comm0);

  // Prepare send/receive displacements
  std::vector<std::int32_t> send_disp = {0};
  std::partial_sum(num_items_per_src.begin(), num_items_per_src.end(),
                   std::back_insert_iterator(send_disp));
  std::vector<std::int32_t> recv_disp = {0};
  std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                   std::back_insert_iterator(recv_disp));

  // Pack my requested indices (global) in send buffer ready to send to
  // post offices
  assert(send_disp.back() == (int)src_to_index.size());
  std::vector<std::int64_t> send_buffer_index(src_to_index.size());
  std::transform(src_to_index.cbegin(), src_to_index.cend(),
                 send_buffer_index.begin(),
                 [](auto& x) { return std::get<1>(x); });

  // Prepare the receive buffer
  std::vector<std::int64_t> recv_buffer_index(recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer_index.data(), num_items_per_src.data(),
                         send_disp.data(), MPI_INT64_T,
                         recv_buffer_index.data(), num_items_recv.data(),
                         recv_disp.data(), MPI_INT64_T, neigh_comm0);

  MPI_Comm_free(&neigh_comm0);

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

  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm0);

  MPI_Datatype compound_type0;
  MPI_Type_contiguous(shape[1], dolfinx::MPI::mpi_type<T>(), &compound_type0);
  MPI_Type_commit(&compound_type0);

  std::vector<T> recv_buffer_data(shape[1] * send_disp.back());
  MPI_Neighbor_alltoallv(send_buffer_data.data(), num_items_recv.data(),
                         recv_disp.data(), compound_type0,
                         recv_buffer_data.data(), num_items_per_src.data(),
                         send_disp.data(), compound_type0, neigh_comm0);

  MPI_Type_free(&compound_type0);
  MPI_Comm_free(&neigh_comm0);

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
                               const xtl::span<const std::int64_t>& indices,
                               const xtl::span<const T>& x, int shape1)
{
  assert(shape1 > 1);
  assert(x.size() % shape1 == 0);
  const std::int64_t shape0_local = x.size() / shape1;

  std::int64_t shape0(0), rank_offset(0);
  MPI_Allreduce(&shape0_local, &shape0, 1, MPI_INT64_T, MPI_SUM, comm);
  MPI_Exscan(&shape0_local, &rank_offset, 1, MPI_INT64_T, MPI_SUM, comm);

  return distribute_from_postoffice(comm, indices, x, {shape0, shape1},
                                    rank_offset);
}
//---------------------------------------------------------------------------
template <typename T>
std::vector<T> distribute_data1(MPI_Comm comm,
                                const xtl::span<const std::int64_t>& indices,
                                const xtl::span<const T>& x, int shape1)
{
  common::Timer timer("Distribute row-wise data (non-scalable)");
  assert(shape1 > 0);

  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  assert(x.size() % shape1 == 0);
  const std::int64_t shape0 = x.size() / shape1;

  // Get number of rows on each rank
  std::vector<std::int64_t> global_sizes(size);
  MPI_Allgather(&shape0, 1, MPI_INT64_T, global_sizes.data(), 1, MPI_INT64_T,
                comm);
  std::vector<std::int64_t> global_offsets(size + 1, 0);
  std::partial_sum(global_sizes.begin(), global_sizes.end(),
                   std::next(global_offsets.begin()));

  // Build index data requests
  std::vector<int> number_index_send(size, 0);
  std::vector<int> index_owner(indices.size());
  std::vector<int> index_order(indices.size());
  std::iota(index_order.begin(), index_order.end(), 0);
  std::sort(index_order.begin(), index_order.end(),
            [&indices](int a, int b) { return (indices[a] < indices[b]); });

  int p = 0;
  for (std::size_t i = 0; i < index_order.size(); ++i)
  {
    int j = index_order[i];
    while (indices[j] >= global_offsets[p + 1])
      ++p;
    index_owner[j] = p;
    number_index_send[p]++;
  }

  // Compute send displacements
  std::vector<int> disp_index_send(size + 1, 0);
  std::partial_sum(number_index_send.begin(), number_index_send.end(),
                   std::next(disp_index_send.begin()));

  // Pack global index send data
  std::vector<std::int64_t> indices_send(disp_index_send.back());
  std::vector<int> disp_tmp = disp_index_send;
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    const int owner = index_owner[i];
    indices_send[disp_tmp[owner]++] = indices[i];
  }

  // Send/receive number of indices to communicate to each process
  std::vector<int> number_index_recv(size);
  MPI_Alltoall(number_index_send.data(), 1, MPI_INT, number_index_recv.data(),
               1, MPI_INT, comm);

  // Compute receive displacements
  std::vector<int> disp_index_recv(size + 1, 0);
  std::partial_sum(number_index_recv.begin(), number_index_recv.end(),
                   std::next(disp_index_recv.begin()));

  // Send/receive global indices
  std::vector<std::int64_t> indices_recv(disp_index_recv.back());
  MPI_Alltoallv(indices_send.data(), number_index_send.data(),
                disp_index_send.data(), MPI_INT64_T, indices_recv.data(),
                number_index_recv.data(), disp_index_recv.data(), MPI_INT64_T,
                comm);

  // Pack point data to send back (transpose)
  std::vector<T> x_return(indices_recv.size() * shape1);
  for (int p = 0; p < size; ++p)
  {
    for (int i = disp_index_recv[p]; i < disp_index_recv[p + 1]; ++i)
    {
      const std::int32_t index_local = indices_recv[i] - global_offsets[rank];
      assert(index_local >= 0);
      std::copy_n(std::next(x.cbegin(), shape1 * index_local), shape1,
                  std::next(x_return.begin(), shape1 * i));
    }
  }

  MPI_Datatype compound_type;
  MPI_Type_contiguous(shape1, dolfinx::MPI::mpi_type<T>(), &compound_type);
  MPI_Type_commit(&compound_type);

  // Send back point data
  std::vector<T> my_x(disp_index_send.back() * shape1);
  MPI_Alltoallv(x_return.data(), number_index_recv.data(),
                disp_index_recv.data(), compound_type, my_x.data(),
                number_index_send.data(), disp_index_send.data(), compound_type,
                comm);
  MPI_Type_free(&compound_type);

  return my_x;
}
//-----------------------------------------------------------------------------
template <typename T>
graph::AdjacencyList<T> all_to_all(MPI_Comm comm,
                                   const graph::AdjacencyList<T>& send_data)
{
  const std::vector<std::int32_t>& send_offsets = send_data.offsets();
  const std::vector<T>& values_in = send_data.array();

  const int comm_size = dolfinx::MPI::size(comm);
  assert(send_data.num_nodes() == comm_size);

  // Data size per destination rank
  std::vector<int> send_size(comm_size);
  std::adjacent_difference(std::next(send_offsets.begin()), send_offsets.end(),
                           send_size.begin());

  // Get received data sizes from each rank
  std::vector<int> recv_size(comm_size);
  MPI_Alltoall(send_size.data(), 1, mpi_type<int>(), recv_size.data(), 1,
               mpi_type<int>(), comm);

  // Compute receive offset
  std::vector<std::int32_t> recv_offset(comm_size + 1, 0);
  std::partial_sum(recv_size.begin(), recv_size.end(),
                   std::next(recv_offset.begin()));

  // Send/receive data
  std::vector<T> recv_values(recv_offset.back());
  MPI_Alltoallv(values_in.data(), send_size.data(), send_offsets.data(),
                mpi_type<T>(), recv_values.data(), recv_size.data(),
                recv_offset.data(), mpi_type<T>(), comm);

  return graph::AdjacencyList<T>(std::move(recv_values),
                                 std::move(recv_offset));
}
//-----------------------------------------------------------------------------
template <typename T>
graph::AdjacencyList<T>
neighbor_all_to_all(MPI_Comm comm, const graph::AdjacencyList<T>& send_data)
{
  // Get neighbor processes
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);

  // Allocate memory (add '1' to handle empty case as OpenMPI fails for
  // null pointers
  std::vector<int> send_sizes(outdegree, 0);
  std::vector<int> recv_sizes(indegree);
  std::adjacent_difference(std::next(send_data.offsets().begin()),
                           send_data.offsets().end(), send_sizes.begin());
  // Get receive sizes
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                        MPI_INT, comm);

  // Work out recv offsets
  std::vector<int> recv_offsets(indegree + 1);
  recv_offsets[0] = 0;
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_offsets.begin(), 1));

  std::vector<T> recv_data(recv_offsets[recv_offsets.size() - 1]);
  MPI_Neighbor_alltoallv(
      send_data.array().data(), send_sizes.data(), send_data.offsets().data(),
      dolfinx::MPI::mpi_type<T>(), recv_data.data(), recv_sizes.data(),
      recv_offsets.data(), dolfinx::MPI::mpi_type<T>(), comm);

  return graph::AdjacencyList<T>(std::move(recv_data), std::move(recv_offsets));
}
//---------------------------------------------------------------------------

} // namespace dolfinx::MPI
