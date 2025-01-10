// Copyright (C) 2021-2022 Garth N. Wells and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "SparsityPattern.h"
#include "Vector.h"
#include "matrix_csr_impl.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <mpi.h>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::la
{
/// @brief Modes for representing block structured matrices
enum class BlockMode : int
{
  compact = 0, /// Each entry in the sparsity pattern of the matrix refers to a
               /// block of data of size (bs[0], bs[1]).
  expanded = 1 /// The sparsity pattern is expanded by (bs[0], bs[1]), and each
               /// matrix entry refers to one data item, i.e. the resulting
               /// matrix has a block size of (1, 1).
};

/// @brief Distributed sparse matrix.
///
/// The matrix storage format is compressed sparse row. The matrix is
/// partitioned row-wise across MPI ranks.
///
/// @warning Experimental storage of a matrix in CSR format which
/// can be assembled into using the usual DOLFINx assembly routines.
/// Matrix internal data can be accessed for interfacing with other
/// code.
///
/// @tparam Scalar Scalar type of matrix entries
/// @tparam Container Sequence container type to store matrix entries
/// @tparam ColContainer Column index container type
/// @tparam RowPtrContainer Row pointer container type
template <class Scalar, class Container = std::vector<Scalar>,
          class ColContainer = std::vector<std::int32_t>,
          class RowPtrContainer = std::vector<std::int64_t>>
class MatrixCSR
{
  static_assert(std::is_same_v<typename Container::value_type, Scalar>);

public:
  /// Scalar type
  using value_type = Scalar;

  /// Matrix entries container type
  using container_type = Container;

  /// Column index container type
  using column_container_type = ColContainer;

  /// Row pointer container type
  using rowptr_container_type = RowPtrContainer;

  static_assert(std::is_same_v<value_type, typename container_type::value_type>,
                "Scalar type and container value type must be the same.");

  /// @brief Insertion functor for setting values in a matrix. It is
  /// typically used in finite element assembly functions.
  ///
  /// Create a function to set values in a MatrixCSR. The function
  /// signature is `int mat_set_fn(std::span<const std::int32_t rows,
  /// std::span<const std::int32_t cols, std::span<const value_type>
  /// data)`. The rows and columns use process local indexing, and the
  /// given rows and columns must pre-exist in the sparsity pattern of
  /// the matrix. Insertion into "ghost" rows (in the ghost region of
  /// the row `IndexMap`) is permitted, so long as there are correct
  /// entries in the sparsity pattern.
  ///
  /// @note Using rows or columns which are not in the sparsity will
  /// result in undefined behaviour (or an assert failure in Debug
  /// mode).
  ///
  /// @note Matrix block size may be (1, 1) or (BS0, BS1)
  /// @note Data block size may be (1, 1) or (BS0, BS1)
  ///
  /// @tparam BS0 Row block size of data for insertion
  /// @tparam BS1 Column block size of data for insertion
  ///
  /// @return Function for inserting values into `A`
  template <int BS0 = 1, int BS1 = 1>
  auto mat_set_values()
  {
    if ((BS0 != _bs[0] and BS0 > 1 and _bs[0] > 1)
        or (BS1 != _bs[1] and BS1 > 1 and _bs[1] > 1))
    {
      throw std::runtime_error(
          "Cannot insert blocks of different size than matrix block size");
    }

    return [&](std::span<const std::int32_t> rows,
               std::span<const std::int32_t> cols,
               std::span<const value_type> data) -> int
    {
      this->set<BS0, BS1>(data, rows, cols);
      return 0;
    };
  }

  /// @brief Insertion functor for adding values to a matrix. It is
  /// typically used in finite element assembly functions.
  ///
  /// Create a function to add values to a MatrixCSR. The function
  /// signature is `int mat_add_fn(std::span<const std::int32_t rows,
  /// std::span<const std::int32_t cols, std::span<const value_type>
  /// data)`. The rows and columns use process local indexing, and the
  /// given rows and columns must pre-exist in the sparsity pattern of
  /// the matrix. Insertion into "ghost" rows (in the ghost region of
  /// the row `IndexMap`) is permitted, so long as there are correct
  /// entries in the sparsity pattern.
  ///
  /// @note Using rows or columns which are not in the sparsity will
  /// result in undefined behaviour (or an assert failure in Debug
  /// mode).
  ///
  /// @note Matrix block size may be (1, 1) or (BS0, BS1)
  /// @note Data block size may be (1, 1) or (BS0, BS1)
  ///
  /// @tparam BS0 Row block size of data for insertion
  /// @tparam BS1 Column block size of data for insertion
  ///
  /// @return Function for inserting values into `A`
  template <int BS0 = 1, int BS1 = 1>
  auto mat_add_values()
  {
    if ((BS0 != _bs[0] and BS0 > 1 and _bs[0] > 1)
        or (BS1 != _bs[1] and BS1 > 1 and _bs[1] > 1))
    {
      throw std::runtime_error(
          "Cannot insert blocks of different size than matrix block size");
    }

    return [&](std::span<const std::int32_t> rows,
               std::span<const std::int32_t> cols,
               std::span<const value_type> data) -> int
    {
      this->add<BS0, BS1>(data, rows, cols);
      return 0;
    };
  }

  /// @brief Create a distributed matrix.
  ///
  /// The structure of the matrix depends entirely on the input
  /// `SparsityPattern`, which must be finalized. The matrix storage is
  /// distributed Compressed Sparse Row: the matrix is distributed by
  /// row across processes, and on each process, there is a list of
  /// column indices and matrix entries for each row stored. This
  /// exactly matches the layout of the `SparsityPattern`. There is some
  /// overlap of matrix rows between processes to allow for independent
  /// Finite Element assembly, after which, the ghost rows should be
  /// sent to the row owning processes by calling `scatter_rev()`.
  ///
  /// @note The block size of the matrix is given by the block size of
  /// the input `SparsityPattern`.
  ///
  /// @param[in] p The sparsity pattern which describes the parallel
  /// distribution and the non-zero structure.
  /// @param[in] mode Block mode. When the block size is greater than
  /// one, the storage can be "compact" where each matrix entry refers
  /// to a block of data (stored row major), or "expanded" where each
  /// matrix entry is individual. In the "expanded" case, the sparsity
  /// is expanded for every entry in the block, and the block size of
  /// the matrix is set to (1, 1).
  MatrixCSR(const SparsityPattern& p, BlockMode mode = BlockMode::compact);

  /// Move constructor
  /// @todo Check handling of MPI_Request
  MatrixCSR(MatrixCSR&& A) = default;

  /// @brief Set all non-zero local entries to a value including entries
  /// in ghost rows.
  /// @param[in] x The value to set non-zero matrix entries to
  void set(value_type x) { std::ranges::fill(_data, x); }

  /// @brief Set values in the matrix.
  ///
  /// @note Only entries included in the sparsity pattern used to
  /// initialize the matrix can be set.
  /// @note All indices are local to the calling MPI rank and entries
  /// cannot be set in ghost rows.
  /// @note This should be called after `scatter_rev`. Using before
  /// `scatter_rev` will set the values correctly, but incoming values
  /// may get added to them during a subsequent reverse scatter
  /// operation.
  /// @tparam BS0 Data row block size
  /// @tparam BS1 Data column block size
  /// @param[in] x The `m` by `n` dense block of values (row-major) to
  /// set in the matrix
  /// @param[in] rows The row indices of `x`
  /// @param[in] cols The column indices of `x`
  template <int BS0, int BS1>
  void set(std::span<const value_type> x, std::span<const std::int32_t> rows,
           std::span<const std::int32_t> cols)
  {
    auto set_fn = [](value_type& y, const value_type& x) { y = x; };

    std::int32_t num_rows
        = _index_maps[0]->size_local() + _index_maps[0]->num_ghosts();
    assert(x.size() == rows.size() * cols.size() * BS0 * BS1);
    if (_bs[0] == BS0 and _bs[1] == BS1)
    {
      impl::insert_csr<BS0, BS1>(_data, _cols, _row_ptr, x, rows, cols, set_fn,
                                 num_rows);
    }
    else if (_bs[0] == 1 and _bs[1] == 1)
    {
      // Set blocked data in a regular CSR matrix (_bs[0]=1, _bs[1]=1)
      // with correct sparsity
      impl::insert_blocked_csr<BS0, BS1>(_data, _cols, _row_ptr, x, rows, cols,
                                         set_fn, num_rows);
    }
    else
    {
      assert(BS0 == 1 and BS1 == 1);
      // Set non-blocked data in a blocked CSR matrix (BS0=1, BS1=1)
      impl::insert_nonblocked_csr(_data, _cols, _row_ptr, x, rows, cols, set_fn,
                                  num_rows, _bs[0], _bs[1]);
    }
  }

  /// @brief Accumulate values in the matrix
  /// @note Only entries included in the sparsity pattern used to
  /// initialize the matrix can be accumulated into.
  /// @note All indices are local to the calling MPI rank and entries
  /// may go into ghost rows.
  /// @note Use `scatter_rev` after all entries have been added to send
  /// ghost rows to owners. Adding more entries after `scatter_rev` is
  /// allowed, but another call to `scatter_rev` will then be required.
  ///
  /// @tparam BS0 Row block size of data
  /// @tparam BS1 Column block size of data
  /// @param[in] x The `m` by `n` dense block of values (row-major) to
  /// add to the matrix
  /// @param[in] rows The row indices of `x`
  /// @param[in] cols The column indices of `x`
  template <int BS0 = 1, int BS1 = 1>
  void add(std::span<const value_type> x, std::span<const std::int32_t> rows,
           std::span<const std::int32_t> cols)
  {
    auto add_fn = [](value_type& y, const value_type& x) { y += x; };

    assert(x.size() == rows.size() * cols.size() * BS0 * BS1);
    if (_bs[0] == BS0 and _bs[1] == BS1)
    {
      impl::insert_csr<BS0, BS1>(_data, _cols, _row_ptr, x, rows, cols, add_fn,
                                 _row_ptr.size());
    }
    else if (_bs[0] == 1 and _bs[1] == 1)
    {
      // Add blocked data to a regular CSR matrix (_bs[0]=1, _bs[1]=1)
      impl::insert_blocked_csr<BS0, BS1>(_data, _cols, _row_ptr, x, rows, cols,
                                         add_fn, _row_ptr.size());
    }
    else
    {
      assert(BS0 == 1 and BS1 == 1);
      // Add non-blocked data to a blocked CSR matrix (BS0=1, BS1=1)
      impl::insert_nonblocked_csr(_data, _cols, _row_ptr, x, rows, cols, add_fn,
                                  _row_ptr.size(), _bs[0], _bs[1]);
    }
  }

  /// Number of local rows excluding ghost rows
  std::int32_t num_owned_rows() const { return _index_maps[0]->size_local(); }

  /// Number of local rows including ghost rows
  std::int32_t num_all_rows() const { return _row_ptr.size() - 1; }

  /// @brief Copy to a dense matrix.
  /// @note This function is typically used for debugging and not used
  /// in production.
  /// @note Ghost rows are also returned, and these can be truncated
  /// manually by using num_owned_rows() if required.
  /// @note If the block size is greater than 1, the entries are
  /// expanded.
  /// @return Dense copy of the part of the matrix on the calling rank.
  /// Storage is row-major.
  std::vector<value_type> to_dense() const;

  /// @brief Transfer ghost row data to the owning ranks accumulating
  /// received values on the owned rows, and zeroing any existing data
  /// in ghost rows.
  ///
  /// This process is analogous to `scatter_rev` for `Vector` except
  /// that the values are always accumulated on the owning process.
  void scatter_rev()
  {
    scatter_rev_begin();
    scatter_rev_end();
  }

  /// @brief Begin transfer of ghost row data to owning ranks, where it
  /// will be accumulated into existing owned rows.
  /// @note Calls to this function must be followed by
  /// MatrixCSR::scatter_rev_end(). Between the two calls matrix values
  /// must not be changed.
  /// @note This function does not change the matrix data. Data update
  /// only occurs with `scatter_rev_end()`.
  void scatter_rev_begin();

  /// @brief End transfer of ghost row data to owning ranks.
  /// @note Must be preceded by MatrixCSR::scatter_rev_begin().
  /// @note Matrix data received from other processes will be
  /// accumulated into locally owned rows, and ghost rows will be
  /// zeroed.
  void scatter_rev_end();

  /// @brief Compute the Frobenius norm squared across all processes.
  /// @note MPI Collective
  double squared_norm() const;

  /// @brief Computes y += Ax for the local CSR matrix and local dense vectors
  ///
  /// @param[in] x Input Vector
  /// @param[out] y Output vector
  void spmv(Vector<value_type>& x, Vector<value_type>& y);

  /// @brief Index maps for the row and column space.
  ///
  /// The row IndexMap contains ghost entries for rows which may be
  /// inserted into and the column IndexMap contains all local and ghost
  /// columns that may exist in the owned rows.
  ///
  /// @return Row (0) or column (1) index maps
  std::shared_ptr<const common::IndexMap> index_map(int dim) const
  {
    return _index_maps.at(dim);
  }

  /// Get local data values
  /// @note Includes ghost values
  container_type& values() { return _data; }

  /// Get local values (const version)
  /// @note Includes ghost values
  const container_type& values() const { return _data; }

  /// Get local row pointers
  /// @note Includes pointers to ghost rows
  const rowptr_container_type& row_ptr() const { return _row_ptr; }

  /// Get local column indices
  /// @note Includes columns in ghost rows
  const column_container_type& cols() const { return _cols; }

  /// Get the start of off-diagonal (unowned columns) on each row,
  /// allowing the matrix to be split (virtually) into two parts.
  /// Operations (such as matrix-vector multiply) between the owned
  /// parts of the matrix and vector can then be performed separately
  /// from operations on the unowned parts.
  /// @note Includes ghost rows, which should be truncated manually if
  /// not required.
  const rowptr_container_type& off_diag_offset() const
  {
    return _off_diagonal_offset;
  }

  /// Block size
  /// @return block sizes for rows and columns
  std::array<int, 2> block_size() const { return _bs; }

private:
  // Maps for the distribution of the ows and columns
  std::array<std::shared_ptr<const common::IndexMap>, 2> _index_maps;

  // Block mode (compact or expanded)
  BlockMode _block_mode;

  // Block sizes
  std::array<int, 2> _bs;

  // Matrix data
  container_type _data;
  column_container_type _cols;
  rowptr_container_type _row_ptr;

  // Start of off-diagonal (unowned columns) on each row
  rowptr_container_type _off_diagonal_offset;

  // Neighborhood communicator (ghost->owner communicator for rows)
  dolfinx::MPI::Comm _comm;

  // -- Precomputed data for scatter_rev/update

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

  // Temporary stores for data during non-blocking communication
  container_type _ghost_value_data;
  container_type _ghost_value_data_in;
};
//-----------------------------------------------------------------------------
template <class U, class V, class W, class X>
MatrixCSR<U, V, W, X>::MatrixCSR(const SparsityPattern& p, BlockMode mode)
    : _index_maps({p.index_map(0),
                   std::make_shared<common::IndexMap>(p.column_index_map())}),
      _block_mode(mode), _bs({p.block_size(0), p.block_size(1)}),
      _data(p.num_nonzeros() * _bs[0] * _bs[1], 0),
      _cols(p.graph().first.begin(), p.graph().first.end()),
      _row_ptr(p.graph().second.begin(), p.graph().second.end()),
      _comm(MPI_COMM_NULL)
{
  if (_block_mode == BlockMode::expanded)
  {
    // Rebuild IndexMaps
    for (int i = 0; i < 2; ++i)
    {
      const auto im = _index_maps[i];
      const int size_local = im->size_local() * _bs[i];
      std::span ghost_i = im->ghosts();
      std::vector<std::int64_t> ghosts;
      const std::vector<int> ghost_owner_i(im->owners().begin(),
                                           im->owners().end());
      std::vector<int> src_rank;
      for (std::size_t j = 0; j < ghost_i.size(); ++j)
      {
        for (int k = 0; k < _bs[i]; ++k)
        {
          ghosts.push_back(ghost_i[j] * _bs[i] + k);
          src_rank.push_back(ghost_owner_i[j]);
        }
      }
      const std::array<std::vector<int>, 2> src_dest0
          = {std::vector(_index_maps[i]->src().begin(),
                         _index_maps[i]->src().end()),
             std::vector(_index_maps[i]->dest().begin(),
                         _index_maps[i]->dest().end())};
      _index_maps[i] = std::make_shared<common::IndexMap>(
          _index_maps[i]->comm(), size_local, src_dest0, ghosts, src_rank);
    }

    // Convert sparsity pattern and set _bs to 1

    column_container_type new_cols;
    new_cols.reserve(_data.size());
    rowptr_container_type new_row_ptr = {0};
    new_row_ptr.reserve(_row_ptr.size() * _bs[0]);
    std::span<const std::int32_t> num_diag_nnz = p.off_diagonal_offsets();
    for (std::size_t i = 0; i < _row_ptr.size() - 1; ++i)
    {
      // Repeat row _bs[0] times
      for (int q0 = 0; q0 < _bs[0]; ++q0)
      {
        _off_diagonal_offset.push_back(new_row_ptr.back()
                                       + num_diag_nnz[i] * _bs[1]);
        for (auto j = _row_ptr[i]; j < _row_ptr[i + 1]; ++j)
        {
          for (int q1 = 0; q1 < _bs[1]; ++q1)
            new_cols.push_back(_cols[j] * _bs[1] + q1);
        }
        new_row_ptr.push_back(new_cols.size());
      }
    }
    _cols = new_cols;
    _row_ptr = new_row_ptr;
    _bs[0] = 1;
    _bs[1] = 1;
  }
  else
  {
    // Compute off-diagonal offset for each row (compact)
    std::span<const std::int32_t> num_diag_nnz = p.off_diagonal_offsets();
    _off_diagonal_offset.reserve(num_diag_nnz.size());
    std::ranges::transform(num_diag_nnz, _row_ptr,
                           std::back_inserter(_off_diagonal_offset),
                           std::plus{});
  }

  // Some short-hand
  const std::array local_size
      = {_index_maps[0]->size_local(), _index_maps[1]->size_local()};
  const std::array local_range
      = {_index_maps[0]->local_range(), _index_maps[1]->local_range()};
  std::span ghosts1 = _index_maps[1]->ghosts();

  std::span ghosts0 = _index_maps[0]->ghosts();
  std::span src_ranks = _index_maps[0]->src();
  std::span dest_ranks = _index_maps[0]->dest();

  // Create neighbourhood communicator (owner <- ghost)
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
    auto it = std::ranges::lower_bound(src_ranks, r);
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
    std::ranges::transform(data_per_proc, std::back_inserter(send_sizes),
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
  const int bs2 = _bs[0] * _bs[1];
  std::ranges::transform(recv_disp, _val_recv_disp.begin(),
                         [&bs2](auto d) { return bs2 * d / 2; });
  std::ranges::transform(_val_send_disp, _val_send_disp.begin(),
                         [&bs2](auto d) { return d * bs2; });

  // Global-to-local map for ghost columns
  std::vector<std::pair<std::int64_t, std::int32_t>> global_to_local;
  global_to_local.reserve(ghosts1.size());
  for (std::int64_t idx : ghosts1)
    global_to_local.push_back({idx, global_to_local.size() + local_size[1]});
  std::ranges::sort(global_to_local);

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
      auto it = std::ranges::lower_bound(
          global_to_local, std::pair(ghost_index_array[i + 1], -1),
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
//-----------------------------------------------------------------------------
template <typename U, typename V, typename W, typename X>
std::vector<typename MatrixCSR<U, V, W, X>::value_type>
MatrixCSR<U, V, W, X>::to_dense() const
{
  const std::size_t nrows = num_all_rows();
  const std::size_t ncols = _index_maps[1]->size_global();
  std::vector<value_type> A(nrows * ncols * _bs[0] * _bs[1], 0.0);
  for (std::size_t r = 0; r < nrows; ++r)
    for (std::int32_t j = _row_ptr[r]; j < _row_ptr[r + 1]; ++j)
      for (int i0 = 0; i0 < _bs[0]; ++i0)
        for (int i1 = 0; i1 < _bs[1]; ++i1)
        {
          std::array<std::int32_t, 1> local_col{_cols[j]};
          std::array<std::int64_t, 1> global_col{0};
          _index_maps[1]->local_to_global(local_col, global_col);
          A[(r * _bs[1] + i0) * ncols * _bs[0] + global_col[0] * _bs[1] + i1]
              = _data[j * _bs[0] * _bs[1] + i0 * _bs[1] + i1];
        }

  return A;
}
//-----------------------------------------------------------------------------
template <typename U, typename V, typename W, typename X>
void MatrixCSR<U, V, W, X>::scatter_rev_begin()
{
  const std::int32_t local_size0 = _index_maps[0]->size_local();
  const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();
  const int bs2 = _bs[0] * _bs[1];

  // For each ghost row, pack and send values to send to neighborhood
  std::vector<int> insert_pos = _val_send_disp;
  _ghost_value_data.resize(_val_send_disp.back());
  for (int i = 0; i < num_ghosts0; ++i)
  {
    const int rank = _ghost_row_to_rank[i];

    // Get position in send buffer to place data to send to this
    // neighbour
    const std::int32_t val_pos = insert_pos[rank];
    std::copy(std::next(_data.data(), _row_ptr[local_size0 + i] * bs2),
              std::next(_data.data(), _row_ptr[local_size0 + i + 1] * bs2),
              std::next(_ghost_value_data.begin(), val_pos));
    insert_pos[rank]
        += bs2 * (_row_ptr[local_size0 + i + 1] - _row_ptr[local_size0 + i]);
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
      _ghost_value_data.data(), val_send_count.data(), _val_send_disp.data(),
      dolfinx::MPI::mpi_t<value_type>, _ghost_value_data_in.data(),
      val_recv_count.data(), _val_recv_disp.data(),
      dolfinx::MPI::mpi_t<value_type>, _comm.comm(), &_request);
  assert(status == MPI_SUCCESS);
}
//-----------------------------------------------------------------------------
template <typename U, typename V, typename W, typename X>
void MatrixCSR<U, V, W, X>::scatter_rev_end()
{
  int status = MPI_Wait(&_request, MPI_STATUS_IGNORE);
  assert(status == MPI_SUCCESS);

  _ghost_value_data.clear();
  _ghost_value_data.shrink_to_fit();

  // Add to local rows
  const int bs2 = _bs[0] * _bs[1];
  assert(_ghost_value_data_in.size() == _unpack_pos.size() * bs2);
  for (std::size_t i = 0; i < _unpack_pos.size(); ++i)
    for (int j = 0; j < bs2; ++j)
      _data[_unpack_pos[i] * bs2 + j] += _ghost_value_data_in[i * bs2 + j];

  _ghost_value_data_in.clear();
  _ghost_value_data_in.shrink_to_fit();

  // Set ghost row data to zero
  const std::int32_t local_size0 = _index_maps[0]->size_local();
  std::fill(std::next(_data.begin(), _row_ptr[local_size0] * bs2), _data.end(),
            0);
}
//-----------------------------------------------------------------------------
template <typename U, typename V, typename W, typename X>
double MatrixCSR<U, V, W, X>::squared_norm() const
{
  const std::size_t num_owned_rows = _index_maps[0]->size_local();
  const int bs2 = _bs[0] * _bs[1];
  assert(num_owned_rows < _row_ptr.size());
  double norm_sq_local = std::accumulate(
      _data.cbegin(), std::next(_data.cbegin(), _row_ptr[num_owned_rows] * bs2),
      double(0), [](auto norm, value_type y) { return norm + std::norm(y); });
  double norm_sq;
  MPI_Allreduce(&norm_sq_local, &norm_sq, 1, MPI_DOUBLE, MPI_SUM, _comm.comm());
  return norm_sq;
}
//-----------------------------------------------------------------------------

// The matrix A is distributed across P  processes by blocks of rows:
//  A = |   A_0  |
//      |   A_1  |
//      |   ...  |
//      |  A_P-1 |
//
// Each submatrix A_i is owned by a single process "i" and can be further
// decomposed into diagonal (Ai[0]) and off diagonal (Ai[1]) blocks:
//  Ai = |Ai[0] Ai[1]|
//
// If A is square, the diagonal block Ai[0] is also square and contains
// only owned columns and rows. The block Ai[1] contains ghost columns
// (unowned dofs).

// Likewise, a local vector x can be decomposed into owned and ghost blocks:
// xi = |   x[0]  |
//      |   x[1]  |
//
// So the product y = Ax can be computed into two separate steps:
//  y[0] = |Ai[0] Ai[1]| |   x[0]  | = Ai[0] x[0] + Ai[1] x[1]
//                       |   x[1]  |
//
/// Computes y += A*x for a parallel CSR matrix A and parallel dense vectors x,y
template <typename Scalar, typename V, typename W, typename X>
void MatrixCSR<Scalar, V, W, X>::spmv(la::Vector<Scalar>& x,
                                      la::Vector<Scalar>& y)
{
  // start communication (update ghosts)
  x.scatter_fwd_begin();

  const std::int32_t nrowslocal = num_owned_rows();
  std::span<const std::int64_t> Arow_ptr(row_ptr().data(), nrowslocal + 1);
  std::span<const std::int32_t> Acols(cols().data(), Arow_ptr[nrowslocal]);
  std::span<const std::int64_t> Aoff_diag_offset(off_diag_offset().data(),
                                                 nrowslocal);
  std::span<const Scalar> Avalues(values().data(), Arow_ptr[nrowslocal]);

  std::span<const Scalar> _x = x.array();
  std::span<Scalar> _y = y.mutable_array();

  std::span<const std::int64_t> Arow_begin(Arow_ptr.data(), nrowslocal);
  std::span<const std::int64_t> Arow_end(Arow_ptr.data() + 1, nrowslocal);

  // First stage:  spmv - diagonal
  // yi[0] += Ai[0] * xi[0]
  if (_bs[1] == 1)
    impl::spmv<Scalar, 1>(Avalues, Arow_begin, Aoff_diag_offset, Acols, _x, _y,
                          _bs[0], 1);
  else
    impl::spmv<Scalar, -1>(Avalues, Arow_begin, Aoff_diag_offset, Acols, _x, _y,
                           _bs[0], _bs[1]);

  // finalize ghost update
  x.scatter_fwd_end();

  // Second stage:  spmv - off-diagonal
  // yi[0] += Ai[1] * xi[1]
  if (_bs[1] == 1)
    impl::spmv<Scalar, 1>(Avalues, Aoff_diag_offset, Arow_end, Acols, _x, _y,
                          _bs[0], 1);
  else
    impl::spmv<Scalar, -1>(Avalues, Aoff_diag_offset, Arow_end, Acols, _x, _y,
                           _bs[0], _bs[1]);
}

} // namespace dolfinx::la
