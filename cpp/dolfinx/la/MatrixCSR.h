// Copyright (C) 2021-2022 Garth N. Wells and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "SparsityPattern.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <mpi.h>
#include <numeric>
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::la
{

namespace impl
{
/// @brief Set data in a CSR matrix
///
/// @param[out] data The CSR matrix data
/// @param[in] cols The CSR column indices
/// @param[in] row_ptr The pointer to the ith row in the CSR data
/// @param[in] x The `m` by `n` dense block of values (row-major) to set
/// in the matrix
/// @param[in] xrows The row indices of `x`
/// @param[in] xcols The column indices of `x`
/// @param[in] local_size The maximum row index that can be set. Used
/// when debugging is own to check that rows beyond a permitted range
/// are not being set.
template <typename U, typename V, typename W, typename X>
void set_csr(U&& data, const V& cols, const V& row_ptr, const W& x,
             const X& xrows, const X& xcols,
             [[maybe_unused]] typename X::value_type local_size)
{
  assert(x.size() == xrows.size() * xcols.size());
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    auto row = xrows[r];
    using T = typename W::value_type;
    const T* xr = x.data() + r * xcols.size();

#ifndef NDEBUG
    if (row >= local_size)
      throw std::runtime_error("Local row out of range");
#endif

    // Columns indices for row
    auto cit0 = std::next(cols.begin(), row_ptr[row]);
    auto cit1 = std::next(cols.begin(), row_ptr[row + 1]);
    for (std::size_t c = 0; c < xcols.size(); ++c)
    {
      // Find position of column index
      auto it = std::lower_bound(cit0, cit1, xcols[c]);
      assert(it != cit1);
      std::size_t d = std::distance(cols.begin(), it);
      assert(d < data.size());
      data[d] = xr[c];
    }
  }
}

/// @brief Add data to a CSR matrix
///
/// @param[out] data The CSR matrix data
/// @param[in] cols The CSR column indices
/// @param[in] row_ptr The pointer to the ith row in the CSR data
/// @param[in] x The `m` by `n` dense block of values (row-major) to add
/// to the matrix
/// @param[in] xrows The row indices of `x`
/// @param[in] xcols The column indices of `x`
template <typename U, typename V, typename W, typename X>
void add_csr(U&& data, const V& cols, const V& row_ptr, const W& x,
             const X& xrows, const X& xcols)
{
  assert(x.size() == xrows.size() * xcols.size());
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    auto row = xrows[r];
    using T = typename W::value_type;
    const T* xr = x.data() + r * xcols.size();

#ifndef NDEBUG
    if (row >= (int)row_ptr.size())
      throw std::runtime_error("Local row out of range");
#endif

    // Columns indices for row
    auto cit0 = std::next(cols.begin(), row_ptr[row]);
    auto cit1 = std::next(cols.begin(), row_ptr[row + 1]);
    for (std::size_t c = 0; c < xcols.size(); ++c)
    {
      // Find position of column index
      auto it = std::lower_bound(cit0, cit1, xcols[c]);
      assert(it != cit1);
      std::size_t d = std::distance(cols.begin(), it);
      assert(d < data.size());
      data[d] += xr[c];
    }
  }
}
} // namespace impl

/// Distributed sparse matrix
///
/// The matrix storage format is compressed sparse row. The matrix is
/// partitioned row-wise across MPI rank.
///
/// @tparam T The data type for the matrix
/// @tparam Allocator The memory allocator type for the data storage
///
/// @note Highly "experimental" storage of a matrix in CSR format which
/// can be assembled into using the usual dolfinx assembly routines
/// Matrix internal data can be accessed for interfacing with other
/// code.
///
/// @todo Handle block sizes
template <typename T, class Allocator = std::allocator<T>>
class MatrixCSR
{
public:
  /// The value type
  using value_type = T;

  /// The allocator type
  using allocator_type = Allocator;

  /// Insertion functor for setting values in matrix. It is typically
  /// used in finite element assembly functions.
  /// @param A Matrix to insert into
  /// @return Function for inserting values into `A`
  /// @todo clarify setting on non-owned enrties
  auto mat_set_values()
  {
    return [&](const xtl::span<const std::int32_t>& rows,
               const xtl::span<const std::int32_t>& cols,
               const xtl::span<const T>& data) -> int
    {
      this->set(data, rows, cols);
      return 0;
    };
  }

  /// Insertion functor for accumulating values in matrix. It is
  /// typically used in finite element assembly functions.
  /// @param A Matrix to insert into
  /// @return Function for inserting values into `A`
  auto mat_add_values()
  {
    return [&](const xtl::span<const std::int32_t>& rows,
               const xtl::span<const std::int32_t>& cols,
               const xtl::span<const T>& data) -> int
    {
      this->add(data, rows, cols);
      return 0;
    };
  }

  /// Create a distributed matrix
  /// @param[in] p The sparsty pattern the describes the parallel
  /// distribution and the non-zero structure
  /// @param[in] alloc The memory allocator for the data storafe
  MatrixCSR(const SparsityPattern& p, const Allocator& alloc = Allocator())
      : _index_maps({p.index_map(0),
                     std::make_shared<common::IndexMap>(p.column_index_map())}),
        _bs({p.block_size(0), p.block_size(1)}),
        _data(p.num_nonzeros(), 0, alloc),
        _cols(p.graph().array().begin(), p.graph().array().end()),
        _row_ptr(p.graph().offsets().begin(), p.graph().offsets().end()),
        _comm(MPI_COMM_NULL)
  {
    // TODO: handle block sizes
    if (_bs[0] > 1 or _bs[1] > 1)
      throw std::runtime_error("Block size not yet supported");

    // Compute off-diagonal offset for each row
    xtl::span<const std::int32_t> num_diag_nnz = p.off_diagonal_offset();
    _off_diagonal_offset.reserve(num_diag_nnz.size());
    std::transform(num_diag_nnz.begin(), num_diag_nnz.end(), _row_ptr.begin(),
                   std::back_inserter(_off_diagonal_offset),
                   std::plus<std::int32_t>());

    // Some short-hand
    const std::array local_size
        = {_index_maps[0]->size_local(), _index_maps[1]->size_local()};
    const std::array local_range
        = {_index_maps[0]->local_range(), _index_maps[1]->local_range()};
    const std::vector<std::int64_t>& ghosts1 = _index_maps[1]->ghosts();

    const std::vector<std::int64_t>& ghosts0 = _index_maps[0]->ghosts();
    const std::vector<int>& src_ranks = _index_maps[0]->src();
    const std::vector<int>& dest_ranks = _index_maps[0]->dest();

    // Create neigbourhood communicator (owner <- ghost)
    MPI_Comm comm;
    MPI_Dist_graph_create_adjacent(_index_maps[0]->comm(), dest_ranks.size(),
                                   dest_ranks.data(), MPI_UNWEIGHTED,
                                   src_ranks.size(), src_ranks.data(),
                                   MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);
    _comm = dolfinx::MPI::Comm(comm, false);

    // Build map from ghost row index position to owning (neighborhood)
    // rank
    _ghost_row_to_rank.reserve(_index_maps[0]->owners().size());
    for (int r : _index_maps[0]->owners())
    {
      auto it = std::lower_bound(src_ranks.begin(), src_ranks.end(), r);
      assert(it != src_ranks.end() and *it == r);
      int pos = std::distance(src_ranks.begin(), it);
      _ghost_row_to_rank.push_back(pos);
    }

    // Compute size of data to send to each neighbor
    std::vector<std::int32_t> data_per_proc(src_ranks.size(), 0);
    for (std::size_t i = 0; i < _ghost_row_to_rank.size(); ++i)
    {
      assert(_ghost_row_to_rank[i] < data_per_proc.size());
      std::size_t pos = local_size[0] + i;
      data_per_proc[_ghost_row_to_rank[i]] += _row_ptr[pos + 1] - _row_ptr[pos];
    }

    // Compute send displacements
    _val_send_disp.resize(src_ranks.size() + 1, 0);
    std::partial_sum(data_per_proc.begin(), data_per_proc.end(),
                     std::next(_val_send_disp.begin()));

    // For each ghost row, pack and send indices to neighborhood
    std::vector<std::int64_t> ghost_index_data(2 * _val_send_disp.back());
    {
      std::vector<int> insert_pos = _val_send_disp;
      for (std::size_t i = 0; i < _ghost_row_to_rank.size(); ++i)
      {
        const int rank = _ghost_row_to_rank[i];
        int row_id = local_size[0] + i;
        for (int j = _row_ptr[row_id]; j < _row_ptr[row_id + 1]; ++j)
        {
          // Get position in send buffer
          const std::int32_t idx_pos = 2 * insert_pos[rank];

          // Pack send data (row, col) as global indices
          ghost_index_data[idx_pos] = ghosts0[i];
          if (std::int32_t col_local = _cols[j]; col_local < local_size[1])
            ghost_index_data[idx_pos + 1] = col_local + local_range[1][0];
          else
            ghost_index_data[idx_pos + 1] = ghosts1[col_local - local_size[1]];

          insert_pos[rank] += 1;
        }
      }
    }

    // Communicate data with neighborhood
    std::vector<std::int64_t> ghost_index_array;
    std::vector<int> recv_disp;
    {
      std::vector<int> send_sizes;
      std::transform(data_per_proc.begin(), data_per_proc.end(),
                     std::back_inserter(send_sizes),
                     [](auto x) { return 2 * x; });

      std::vector<int> recv_sizes(dest_ranks.size());
      send_sizes.reserve(1);
      recv_sizes.reserve(1);
      MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                            MPI_INT, _comm.comm());

      // Build send/recv displacement
      std::vector<int> send_disp = {0};
      std::partial_sum(send_sizes.begin(), send_sizes.end(),
                       std::back_inserter(send_disp));
      recv_disp = {0};
      std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                       std::back_inserter(recv_disp));

      ghost_index_array.resize(recv_disp.back());
      MPI_Neighbor_alltoallv(ghost_index_data.data(), send_sizes.data(),
                             send_disp.data(), MPI_INT64_T,
                             ghost_index_array.data(), recv_sizes.data(),
                             recv_disp.data(), MPI_INT64_T, _comm.comm());
    }

    // Store receive displacements for future use, when transferring
    // data values
    _val_recv_disp.resize(recv_disp.size());
    std::transform(recv_disp.begin(), recv_disp.end(), _val_recv_disp.begin(),
                   [](int d) { return d / 2; });

    // Global-to-local map for ghost columns
    std::vector<std::pair<std::int64_t, std::int32_t>> global_to_local;
    global_to_local.reserve(ghosts1.size());
    std::int32_t local_i = local_size[1];
    for (std::int64_t idx : ghosts1)
      global_to_local.push_back({idx, global_to_local.size() + local_size[1]});
    std::sort(global_to_local.begin(), global_to_local.end());

    // Compute location in which data for each index should be stored
    // when received
    for (std::size_t i = 0; i < ghost_index_array.size(); i += 2)
    {
      // Row must be on this process
      const std::int32_t local_row = ghost_index_array[i] - local_range[0][0];
      assert(local_row >= 0 and local_row < local_size[0]);

      // Column may be owned or unowned
      std::int32_t local_col = ghost_index_array[i + 1] - local_range[1][0];
      if (local_col < 0 or local_col >= local_size[1])
      {
        const auto it = std::lower_bound(
            global_to_local.begin(), global_to_local.end(),
            std::pair(ghost_index_array[i + 1], -1),
            [](auto& a, auto& b) { return a.first < b.first; });
        assert(it != global_to_local.end()
               and it->first == ghost_index_array[i + 1]);
        local_col = it->second;
      }
      auto cit0 = std::next(_cols.begin(), _row_ptr[local_row]);
      auto cit1 = std::next(_cols.begin(), _row_ptr[local_row + 1]);

      // Find position of column index and insert data
      auto cit = std::lower_bound(cit0, cit1, local_col);
      assert(cit != cit1);
      assert(*cit == local_col);
      std::size_t d = std::distance(_cols.begin(), cit);
      _unpack_pos.push_back(d);
    }
  }

  /// Move constructor
  /// @todo Check handling of MPI_Request
  MatrixCSR(MatrixCSR&& A) = default;

  /// Set all non-zero local entries to a value
  /// including entries in ghost rows
  /// @param[in] x The value to set non-zero matrix entries to
  void set(T x) { std::fill(_data.begin(), _data.end(), x); }

  /// Set values in the matrix
  /// @note Only entries included in the sparsity pattern used to
  /// initialize the matrix can be set
  /// @note All indices are local to the calling MPI rank and entries
  /// cannot be set in ghost rows.
  /// @note This should be called after `finalize`. Using before
  /// `finalize` will set the values correctly, but incoming values may
  /// get added to them during a subsequent finalize operation.
  /// @param[in] x The `m` by `n` dense block of values (row-major) to
  /// set in the matrix
  /// @param[in] rows The row indices of `x`
  /// @param[in] cols The column indices of `x`
  void set(const xtl::span<const T>& x,
           const xtl::span<const std::int32_t>& rows,
           const xtl::span<const std::int32_t>& cols)
  {
    impl::set_csr(_data, _cols, _row_ptr, x, rows, cols,
                  _index_maps[0]->size_local());
  }

  /// Accumulate values in the matrix
  /// @note Only entries included in the sparsity pattern used to
  /// initialize the matrix can be accumulated in to
  /// @note All indices are local to the calling MPI rank and entries
  /// may go into ghost rows.
  /// @note Use `finalize` after all entries have been added to send
  /// ghost rows to owners. Adding more entries after `finalize` is
  /// allowed, but another call to `finalize` will then be required.
  /// @param[in] x The `m` by `n` dense block of values (row-major) to
  /// add to the matrix
  /// @param[in] rows The row indices of `x`
  /// @param[in] cols The column indices of `x`
  void add(const xtl::span<const T>& x,
           const xtl::span<const std::int32_t>& rows,
           const xtl::span<const std::int32_t>& cols)
  {
    impl::add_csr(_data, _cols, _row_ptr, x, rows, cols);
  }

  /// Number of local rows excluding ghost rows
  std::int32_t num_owned_rows() const { return _index_maps[0]->size_local(); }

  /// Number of local rows including ghost rows
  std::int32_t num_all_rows() const { return _row_ptr.size() - 1; }

  /// Copy to a dense matrix
  /// @note This function is typically used for debugging and not used
  /// in production
  /// @note Ghost rows are also returned, and these can be truncated
  /// manually by using num_owned_rows() if required.
  /// @return Dense copy of the part of the matrix on the calling rank.
  /// Storage is row-major.
  std::vector<T> to_dense() const
  {
    const std::size_t nrows = num_all_rows();
    const std::size_t ncols
        = _index_maps[1]->size_local() + _index_maps[1]->num_ghosts();
    std::vector<T> A(nrows * ncols);
    for (std::size_t r = 0; r < nrows; ++r)
      for (std::int32_t j = _row_ptr[r]; j < _row_ptr[r + 1]; ++j)
        A[r * ncols + _cols[j]] = _data[j];

    return A;
  }

  /// Transfer ghost row data to the owning ranks
  /// accumulating received values on the owned rows, and zeroing any existing
  /// data in ghost rows.
  void finalize()
  {
    finalize_begin();
    finalize_end();
  }

  /// Begin transfer of ghost row data to owning ranks, where it will be
  /// accumulated into existing owned rows.
  /// @note Calls to this function must be followed by
  /// MatrixCSR::finalize_end(). Between the two calls matrix values
  /// must not be changed.
  /// @note This function does not change the matrix data. Data update only
  /// occurs with `finalize_end()`.
  void finalize_begin()
  {
    const std::int32_t local_size0 = _index_maps[0]->size_local();
    const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();

    // For each ghost row, pack and send values to send to neighborhood
    std::vector<int> insert_pos = _val_send_disp;
    std::vector<T> ghost_value_data(_val_send_disp.back());
    for (int i = 0; i < num_ghosts0; ++i)
    {
      const int rank = _ghost_row_to_rank[i];

      // Get position in send buffer to place data to send to this
      // neighbour
      const std::int32_t val_pos = insert_pos[rank];
      std::copy(std::next(_data.data(), _row_ptr[local_size0 + i]),
                std::next(_data.data(), _row_ptr[local_size0 + i + 1]),
                std::next(ghost_value_data.begin(), val_pos));
      insert_pos[rank]
          += _row_ptr[local_size0 + i + 1] - _row_ptr[local_size0 + i];
    }

    _ghost_value_data_in.resize(_val_recv_disp.back());

    // Compute data sizes for send and receive from displacements
    std::vector<int> val_send_count(_val_send_disp.size() - 1);
    std::adjacent_difference(std::next(_val_send_disp.begin()),
                             _val_send_disp.end(), val_send_count.begin());

    std::vector<int> val_recv_count(_val_recv_disp.size() - 1);
    std::adjacent_difference(std::next(_val_recv_disp.begin()),
                             _val_recv_disp.end(), val_recv_count.begin());

    int status = MPI_Ineighbor_alltoallv(
        ghost_value_data.data(), val_send_count.data(), _val_send_disp.data(),
        dolfinx::MPI::mpi_type<T>(), _ghost_value_data_in.data(),
        val_recv_count.data(), _val_recv_disp.data(),
        dolfinx::MPI::mpi_type<T>(), _comm.comm(), &_request);
    assert(status == MPI_SUCCESS);
  }

  /// End transfer of ghost row data to owning ranks
  /// @note Must be preceded by MatrixCSR::finalize_begin()
  /// @note Matrix data received from other processes will be
  /// accumulated into locally owned rows, and ghost rows will be
  /// zeroed.
  void finalize_end()
  {
    int status = MPI_Wait(&_request, MPI_STATUS_IGNORE);
    assert(status == MPI_SUCCESS);

    // Add to local rows
    assert(_ghost_value_data_in.size() == _unpack_pos.size());
    for (std::size_t i = 0; i < _ghost_value_data_in.size(); ++i)
      _data[_unpack_pos[i]] += _ghost_value_data_in[i];

    // Set ghost row data to zero
    const std::int32_t local_size0 = _index_maps[0]->size_local();
    std::fill(std::next(_data.begin(), _row_ptr[local_size0]), _data.end(), 0);
  }

  /// Compute the Frobenius norm squared
  double norm_squared() const
  {
    const std::size_t num_owned_rows = _index_maps[0]->size_local();
    assert(num_owned_rows < _row_ptr.size());

    const double norm_sq_local = std::accumulate(
        _data.cbegin(), std::next(_data.cbegin(), _row_ptr[num_owned_rows]),
        double(0), [](double norm, T y) { return norm + std::norm(y); });
    double norm_sq;
    MPI_Allreduce(&norm_sq_local, &norm_sq, 1, MPI_DOUBLE, MPI_SUM,
                  _comm.comm());
    return norm_sq;
  }

  /// Index maps for the row and column space. The row IndexMap contains
  /// ghost entries for rows which may be inserted into and the column
  /// IndexMap contains all local and ghost columns that may exist in
  /// the owned rows.
  ///
  /// @return Row (0) and column (1) index maps
  const std::array<std::shared_ptr<const common::IndexMap>, 2>&
  index_maps() const
  {
    return _index_maps;
  }

  /// Get local data values
  /// @note Includes ghost values
  std::vector<T>& values() { return _data; }

  /// Get local values (const version)
  /// @note Includes ghost values
  const std::vector<T>& values() const { return _data; }

  /// Get local row pointers
  /// @note Includes pointers to ghost rows
  const std::vector<std::int32_t>& row_ptr() const { return _row_ptr; }

  /// Get local column indices
  /// @note Includes columns in ghost rows
  const std::vector<std::int32_t>& cols() const { return _cols; }

  /// Get the start of off-diagonal (unowned columns) on each row,
  /// allowing the matrix to be split (virtually) into two parts.
  /// Operations (such as matrix-vector multiply) between the owned
  /// parts of the matrix and vector can then be performed separately
  /// from operations on the unowned parts.
  /// @note Includes ghost rows, which should be truncated manually if
  /// not required.
  const std::vector<std::int32_t>& off_diag_offset() const
  {
    return _off_diagonal_offset;
  }

private:
  // Maps for the distribution of the ows and columns
  std::array<std::shared_ptr<const common::IndexMap>, 2> _index_maps;

  // Block sizes
  std::array<int, 2> _bs;

  // Matrix data
  std::vector<T, Allocator> _data;
  std::vector<std::int32_t> _cols, _row_ptr;

  // Start of off-diagonal (unowned columns) on each row
  std::vector<std::int32_t> _off_diagonal_offset;

  // Neighborhood communicator (ghost->owner communicator for rows)
  dolfinx::MPI::Comm _comm;

  // -- Precomputed data for finalize/update

  // Request in non-blocking communication
  MPI_Request _request;

  // Position in _data to add received data
  std::vector<int> _unpack_pos;

  // Displacements for alltoall for each neighbor when sending and
  // receiving
  std::vector<int> _val_send_disp, _val_recv_disp;

  // Ownership of each row, by neighbor (for the neighbourhood defined
  // on _comm)
  std::vector<int> _ghost_row_to_rank;

  // Temporary store for finalize data during non-blocking communication
  std::vector<T> _ghost_value_data_in;
};

} // namespace dolfinx::la
