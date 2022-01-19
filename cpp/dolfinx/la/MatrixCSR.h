// Copyright (C) 2021-2022 Garth N. Wells and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "SparsityPattern.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx::la
{

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
  static std::function<int(int nr, const int* r, int nc, const int* c,
                           const T* data)>
  mat_set_values(MatrixCSR& A)
  {
    return [&A](int nr, const int* r, int nc, const int* c, const T* data)
    {
      A.set(tcb::span<const T>(data, nr * nc), tcb::span<const int>(r, nr),
            tcb::span<const int>(c, nc));
      return 0;
    };
  }

  /// Insertion functor for accumulating values in matrix. It is
  /// typically used in finite element assembly functions.
  /// @param A Matrix to insert into
  /// @return Function for inserting values into `A`
  static std::function<int(int nr, const int* r, int nc, const int* c,
                           const T* data)>
  mat_add_values(MatrixCSR& A)
  {
    return [&A](int nr, const int* r, int nc, const int* c, const T* data)
    {
      A.add(tcb::span<const T>(data, nr * nc), tcb::span<const int>(r, nr),
            tcb::span<const int>(c, nc));
      return 0;
    };
  }

  /// Create a distributed matrix
  /// @param[in] p The sparsty pattern the describes the parallel
  /// distribution and the non-zero structure
  /// @param[in] alloc The memory allocator for the data storafe
  MatrixCSR(const SparsityPattern& p, const Allocator& alloc = Allocator())
      : _index_maps({p.index_map(0), p.column_index_map()}),
        _bs({p.block_size(0), p.block_size(1)}),
        _data(p.num_nonzeros(), 0, alloc),
        _cols(p.graph().array().begin(), p.graph().array().end()),
        _row_ptr(p.graph().offsets().begin(), p.graph().offsets().end()),
        _comm(p.index_map(0)->comm(common::IndexMap::Direction::reverse))
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

    // Precompute some data for ghost updates via MPI
    const std::array local_size
        = {_index_maps[0]->size_local(), _index_maps[1]->size_local()};
    const std::array local_range
        = {_index_maps[0]->local_range(), _index_maps[1]->local_range()};
    const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();
    const std::vector<int> ghost_owners0 = _index_maps[0]->ghost_owner_rank();
    const std::vector<std::int64_t>& ghosts0 = _index_maps[0]->ghosts();
    const std::vector<std::int64_t>& ghosts1 = _index_maps[1]->ghosts();

    const std::vector<int> dest_ranks
        = dolfinx::MPI::neighbors(_comm.comm())[1];
    const int num_neighbors = dest_ranks.size();

    // Global-to-local ranks for neighborhood
    std::map<int, std::int32_t> dest_proc_to_neighbor;
    for (std::size_t i = 0; i < dest_ranks.size(); ++i)
      dest_proc_to_neighbor.insert({dest_ranks[i], i});

    // Ownership of each ghost row using neighbor rank
    _ghost_row_to_neighbor_rank.resize(num_ghosts0, -1);
    for (int i = 0; i < num_ghosts0; ++i)
    {
      const auto it = dest_proc_to_neighbor.find(ghost_owners0[i]);
      assert(it != dest_proc_to_neighbor.end());
      _ghost_row_to_neighbor_rank[i] = it->second;
    }

    // Compute size of data to send to each neighbor
    std::vector<std::int32_t> data_per_proc(num_neighbors, 0);
    for (int i = 0; i < num_ghosts0; ++i)
    {
      assert(_ghost_row_to_neighbor_rank[i] < (int)data_per_proc.size());
      data_per_proc[_ghost_row_to_neighbor_rank[i]]
          += _row_ptr[local_size[0] + i + 1] - _row_ptr[local_size[0] + i];
    }

    // Compute send displacements for values and indices (x2)
    _val_send_disp.resize(num_neighbors + 1, 0);
    std::partial_sum(data_per_proc.begin(), data_per_proc.end(),
                     std::next(_val_send_disp.begin()));

    std::vector<int> index_send_disp(num_neighbors + 1);
    std::transform(_val_send_disp.begin(), _val_send_disp.end(),
                   index_send_disp.begin(), [](int d) { return d * 2; });

    // For each ghost row, pack and send indices to neighborhood
    std::vector<int> insert_pos(index_send_disp);
    std::vector<std::int64_t> ghost_index_data(index_send_disp.back());
    for (int i = 0; i < num_ghosts0; ++i)
    {
      const int neighbor_rank = _ghost_row_to_neighbor_rank[i];
      int row_id = local_size[0] + i;
      for (int j = _row_ptr[row_id]; j < _row_ptr[row_id + 1]; ++j)
      {
        // Get index position in send buffer
        const std::int32_t idx_pos = insert_pos[neighbor_rank];

        // Pack send data (row, col) as global indices
        ghost_index_data[idx_pos] = ghosts0[i];
        if (const std::int32_t col_local = _cols[j]; col_local < local_size[1])
          ghost_index_data[idx_pos + 1] = col_local + local_range[1][0];
        else
          ghost_index_data[idx_pos + 1] = ghosts1[col_local - local_size[1]];

        insert_pos[neighbor_rank] += 2;
      }
    }

    // Create and communicate adjacency list to neighborhood
    const graph::AdjacencyList<std::int64_t> ghost_index_data_out(
        std::move(ghost_index_data), std::move(index_send_disp));
    const graph::AdjacencyList<std::int64_t> ghost_index_data_in
        = dolfinx::MPI::neighbor_all_to_all(_comm.comm(), ghost_index_data_out);

    // Store received offsets for future use, when transferring data values.
    _val_recv_disp.resize(ghost_index_data_in.offsets().size());
    std::transform(ghost_index_data_in.offsets().begin(),
                   ghost_index_data_in.offsets().end(), _val_recv_disp.begin(),
                   [](int d) { return d / 2; });

    // Global to local map for ghost columns
    std::map<std::int64_t, std::int32_t> global_to_local;
    std::int32_t local_i = local_size[1];
    for (std::int64_t global_i : ghosts1)
      global_to_local.insert({global_i, local_i++});

    // Compute location in which data for each index should be stored
    // when received
    const std::vector<std::int64_t>& ghost_index_array
        = ghost_index_data_in.array();
    for (std::size_t i = 0; i < ghost_index_array.size(); i += 2)
    {
      // Row must be on this process
      const std::int32_t local_row = ghost_index_array[i] - local_range[0][0];
      assert(local_row >= 0 and local_row < local_size[0]);

      // Column may be owned or unowned
      std::int32_t local_col = ghost_index_array[i + 1] - local_range[1][0];
      if (local_col < 0 or local_col >= local_size[1])
      {
        const auto it = global_to_local.find(ghost_index_array[i + 1]);
        assert(it != global_to_local.end());
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
  /// @param[in] x The value to set non-zero matrix entries to
  /// @todo This should probably also set ghost rows
  void set(T x) { std::fill_n(_data.begin(), _index_maps[0]->size_local(), x); }

  /// Set values in the matrix
  /// @note Only entries included in the sparsity pattern used to
  /// initialize the matrix can be set
  /// @note All indices are local to the calling MPI rank
  /// @param[in] x The `m` by `n` dense block of values (row-major) to
  /// set in the matrix
  /// @param[in] rows The row indices of `x`
  /// @param[in] cols The column indices of `x`
  void set(const xtl::span<const T>& x,
           const xtl::span<const std::int32_t>& rows,
           const xtl::span<const std::int32_t>& cols)
  {
    assert(x.size() == rows.size() * cols.size());
    const std::int32_t local_size0 = _index_maps[0]->size_local();
    const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();
    for (std::size_t r = 0; r < rows.size(); ++r)
    {
      // Columns indices for row
      std::int32_t row = rows[r];

      // Current data row
      const T* xr = x.data() + r * cols.size();

      if (row < local_size0)
      {
        auto cit0 = std::next(_cols.begin(), _row_ptr[row]);
        auto cit1 = std::next(_cols.begin(), _row_ptr[row + 1]);
        for (std::size_t c = 0; c < cols.size(); ++c)
        {
          // Find position of column index
          auto it = std::lower_bound(cit0, cit1, cols[c]);
          assert(it != cit1);
          assert(*it == cols[c]);
          std::size_t d = std::distance(_cols.begin(), it);
          _data[d] = xr[c];
        }
      }
      else
        throw std::runtime_error("Local row out of range");
    }
  }

  /// Accumulate values in the matrix
  /// @note Only entries included in the sparsity pattern used to
  /// initialize the matrix can be accumulated in to
  /// @note All indices are local to the calling MPI rank
  /// @param[in] x The `m` by `n` dense block of values (row-major) to
  /// add to the matrix
  /// @param[in] rows The row indices of `x`
  /// @param[in] cols The column indices of `x`
  void add(const xtl::span<const T>& x,
           const xtl::span<const std::int32_t>& rows,
           const xtl::span<const std::int32_t>& cols)
  {
    assert(x.size() == rows.size() * cols.size());
    const std::int32_t local_size0 = _index_maps[0]->size_local();
    const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();
    for (std::size_t r = 0; r < rows.size(); ++r)
    {
      // Columns indices for row
      std::int32_t row = rows[r];

      // Current data row
      const T* xr = x.data() + r * cols.size();

      if (row < local_size0 + num_ghosts0)
      {
        auto cit0 = std::next(_cols.begin(), _row_ptr[row]);
        auto cit1 = std::next(_cols.begin(), _row_ptr[row + 1]);
        for (std::size_t c = 0; c < cols.size(); ++c)
        {
          // Find position of column index
          auto it = std::lower_bound(cit0, cit1, cols[c]);
          assert(it != cit1);
          assert(*it == cols[c]);
          std::size_t d = std::distance(_cols.begin(), it);
          _data[d] += xr[c];
        }
      }
      else
        throw std::runtime_error("Local row out of range");
    }
  }

  /// Get the number of local rows
  /// @param[in] ghost_rows Set to true to include ghost rows in the
  /// number of local rows
  std::int32_t num_rows(bool ghost_rows = false) const
  {
    return ghost_rows ? _row_ptr.size() - 1 : _index_maps[0]->size_local();
  }

  /// Copy to a dense matrix
  /// @note This function is typically used for debugging and not used
  /// in production
  /// @param[in] ghost_rows Set to true to include ghost rows in the
  /// returned matrix
  /// @return Dense copy of the matrix
  xt::xtensor<T, 2> to_dense(bool ghost_rows = false) const
  {
    const std::int32_t nrows = num_rows(ghost_rows);
    const std::int32_t ncols
        = _index_maps[1]->size_local() + _index_maps[1]->num_ghosts();
    xt::xtensor<T, 2> A = xt::zeros<T>({nrows, ncols});
    for (std::size_t r = 0; r < nrows; ++r)
      for (int j = _row_ptr[r]; j < _row_ptr[r + 1]; ++j)
        A(r, _cols[j]) = _data[j];
    return A;
  }

  /// Communicate ghost row data to the owning ranks
  void finalize()
  {
    finalize_begin();
    finalize_end();
  }

  /// Begin communication of ghost row data to owning ranks
  /// @note Calls to this function must be followed by
  /// MatrixCSR::finalize_end(). Between the two calls matrix values
  /// must not be changed.
  void finalize_begin()
  {
    const std::int32_t local_size0 = _index_maps[0]->size_local();
    const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();

    // For each ghost row, pack and send values to send to neighborhood
    std::vector<int> insert_pos(_val_send_disp);
    std::vector<T> ghost_value_data(_val_send_disp.back());
    for (int i = 0; i < num_ghosts0; ++i)
    {
      const int neighbor_rank = _ghost_row_to_neighbor_rank[i];

      // Get position in send buffer
      const std::int32_t val_pos = insert_pos[neighbor_rank];
      std::copy(std::next(_data.data(), _row_ptr[local_size0 + i]),
                std::next(_data.data(), _row_ptr[local_size0 + i + 1]),
                std::next(ghost_value_data.begin(), val_pos));
      insert_pos[neighbor_rank]
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

  /// Begin communication of ghost row data to owning ranks
  /// @note Must be preceded by MatrixCSR::finalize_begin()
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
  /// @param[in] ghost_rows Set to true to include data of ghost rows
  xtl::span<T> values(bool ghost_rows = false)
  {
    const std::int32_t nrows = num_rows(ghost_rows);
    return xtl::span<T>(_data.data(), _row_ptr.at(nrows));
  }

  /// Get local values (const version)
  /// @param[in] ghost_rows Set to true to include data of ghost rows
  xtl::span<const T> values(bool ghost_rows = false) const
  {
    const std::int32_t nrows = num_rows(ghost_rows);
    return xtl::span<const T>(_data.data(), _row_ptr.at(nrows));
  }

  /// Get local row pointers
  /// @param[in] ghost_rows Set to true to include data of ghost rows
  xtl::span<const std::int32_t> row_ptr(bool ghost_rows = false) const
  {
    const std::int32_t nrows = num_rows(ghost_rows);
    return xtl::span<const std::int32_t>(_row_ptr.data(), nrows + 1);
  }

  /// Get local column indices
  /// @param[in] ghost_rows Set to true to include data of ghost rows
  xtl::span<const std::int32_t> cols(bool ghost_rows = false) const
  {
    const std::int32_t nrows = num_rows(ghost_rows);
    return xtl::span<const std::int32_t>(_cols.data(), _row_ptr.at(nrows));
  }

  /// Get start of off-diagonal (unowned columns) on each row
  /// @param[in] ghost_rows Set to true to include data of ghost rows
  xtl::span<const std::int32_t> off_diag_offset(bool ghost_rows = false) const
  {
    const std::size_t nrows = num_rows(ghost_rows);
    return xtl::span<const std::int32_t>(_off_diagonal_offset.data(), nrows);
  }

private:
  // Maps describing the data layout for rows and columns
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

  // Displacements for alltoall for each neighbor when sending and receiving

  std::vector<int> _val_send_disp, _val_recv_disp;

  // Ownership of each row, by neighbor
  std::vector<int> _ghost_row_to_neighbor_rank;

  // Temporary store for finalize data during non-blocking communication
  std::vector<T> _ghost_value_data_in;
};

} // namespace dolfinx::la
