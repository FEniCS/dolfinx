// Copyright (C) 2021-2022 Garth N. Wells and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "SparsityPattern.h"
#include <dolfinx/graph/AdjacencyList.h>
#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx::la
{

/// Distributed sparse matrix
/// Highly "experimental" storage of a matrix in CSR format
/// which can be assembled into using the usual dolfinx assembly routines
/// Matrix internal data can be accessed for interfacing with other code.
template <typename T, class Allocator = std::allocator<T>>
class MatrixCSR
{
public:
  /// The value type
  using value_type = T;

  /// Create a distributed matrix
  MatrixCSR(const SparsityPattern& p, const Allocator& alloc = Allocator())
      : _index_maps({p.index_map(0), p.column_index_map()}),
        _bs({p.block_size(0), p.block_size(1)}),
        _data(p.num_nonzeros(), 0, alloc),
        _cols(p.graph().array().begin(), p.graph().array().end()),
        _row_ptr(p.graph().offsets().begin(), p.graph().offsets().end())
  {
    // TODO: handle block sizes
    if (_bs[0] > 1 or _bs[1] > 1)
      throw std::runtime_error("Block size not yet supported");

    // Precompute some data for ghost updates via MPI
    // Get ghost->owner communicator for rows
    _neighbor_comm = _index_maps[0]->comm(common::IndexMap::Direction::reverse);

    const std::int32_t local_size0 = _index_maps[0]->size_local();
    const std::array local_range0 = _index_maps[0]->local_range();
    const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();
    const std::vector<std::int64_t>& ghosts0 = _index_maps[0]->ghosts();
    const std::vector<int> ghost_owners0 = _index_maps[0]->ghost_owner_rank();
    const std::int32_t local_size1 = _index_maps[1]->size_local();
    const std::array local_range1 = _index_maps[1]->local_range();
    const std::vector<std::int64_t>& ghosts1 = _index_maps[1]->ghosts();

    const auto dest_ranks = dolfinx::MPI::neighbors(_neighbor_comm)[1];
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
          += _row_ptr[local_size0 + i + 1] - _row_ptr[local_size0 + i];
    }

    // Compute send displacements for values and indices (x2)
    _val_send_disp.resize(num_neighbors + 1, 0);
    std::partial_sum(data_per_proc.begin(), data_per_proc.end(),
                     std::next(_val_send_disp.begin(), 1));

    std::vector<int> index_send_disp(num_neighbors + 1);
    std::transform(_val_send_disp.begin(), _val_send_disp.end(),
                   index_send_disp.begin(), [](int d) { return d * 2; });

    // For each ghost row, pack and send indices to neighborhood
    std::vector<int> insert_pos(index_send_disp);
    std::vector<std::int64_t> ghost_index_data(index_send_disp.back());
    for (int i = 0; i < num_ghosts0; ++i)
    {
      const int neighbor_rank = _ghost_row_to_neighbor_rank[i];
      for (int j = _row_ptr[local_size0 + i]; j < _row_ptr[local_size0 + i + 1];
           ++j)
      {
        // Get index position in send buffer
        const std::int32_t idx_pos = insert_pos[neighbor_rank];

        // Pack send data (row, col) as global indices
        ghost_index_data[idx_pos] = ghosts0[i];
        const std::int32_t col_local = _cols[j];
        if (col_local < local_size1)
          ghost_index_data[idx_pos + 1] = col_local + local_range1[0];
        else
          ghost_index_data[idx_pos + 1] = ghosts1[col_local - local_size1];

        insert_pos[neighbor_rank] += 2;
      }
    }

    // Create and communicate adjacency list to neighborhood
    const graph::AdjacencyList<std::int64_t> ghost_index_data_out(
        std::move(ghost_index_data), std::move(index_send_disp));
    const graph::AdjacencyList<std::int64_t> ghost_index_data_in
        = MPI::neighbor_all_to_all(_neighbor_comm, ghost_index_data_out);

    // Store received offsets for future use, when transferring data values.
    _val_recv_disp.resize(ghost_index_data_in.offsets().size());
    std::transform(ghost_index_data_in.offsets().begin(),
                   ghost_index_data_in.offsets().end(), _val_recv_disp.begin(),
                   [](int d) { return d / 2; });

    // Global to local map for ghost columns
    std::map<std::int64_t, std::int32_t> global_to_local;
    std::int32_t local_i = local_size1;
    for (std::int64_t global_i : ghosts1)
      global_to_local.insert({global_i, local_i++});

    // Compute location in which data for each index should be stored when
    // received
    const std::vector<std::int64_t>& ghost_index_array
        = ghost_index_data_in.array();
    for (std::size_t i = 0; i < ghost_index_array.size(); i += 2)
    {
      // Row must be on this process
      const std::int32_t local_row = ghost_index_array[i] - local_range0[0];
      assert(local_row >= 0 and local_row < local_size0);

      // Column may be owned or unowned
      std::int32_t local_col = ghost_index_array[i + 1] - local_range1[0];
      if (local_col < 0 or local_col >= local_size1)
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

  /// Set all non-zero local entries to a value
  void set(T x)
  {
    const std::int32_t local_size0 = _index_maps[0]->size_local();
    std::fill(_data.begin(), std::next(_data.begin(), _row_ptr[local_size0]),
              x);
  }

  /// Set values in local matrix
  /// @param[in] x The `m` by `n` dense block of values (row-major) to
  /// set in the matrix
  /// @param[in] rows The row indices of `x` (indices are local to the MPI rank)
  /// @param[in] cols The column indices of `x` (indices are local to
  /// the MPI rank)
  void set(const xtl::span<const T>& x,
           const xtl::span<const std::int32_t>& rows,
           const xtl::span<const std::int32_t>& cols)
  {
    const std::int32_t local_size0 = _index_maps[0]->size_local();
    const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();

    assert(x.size() == rows.size() * cols.size());
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
      {
        throw std::runtime_error("Local row out of range");
      }
    }
  }

  /// Insertion functor to set local values in matrix
  /// @param A Matrix to insert into
  static std::function<int(int nr, const int* r, int nc, const int* c,
                           const T* data)>
  mat_set_values(MatrixCSR& A)
  {
    return [&A](int nr, const int* r, int nc, const int* c, const T* data) {
      A.set(tcb::span<const T>(data, nr * nc), tcb::span<const int>(r, nr),
            tcb::span<const int>(c, nc));
      return 0;
    };
  }

  /// Add values to local matrix
  /// @param[in] x The `m` by `n` dense block of values (row-major) to
  /// add to the matrix
  /// @param[in] rows The row indices of `x` (indices are local to the MPI rank)
  /// @param[in] cols The column indices of `x` (indices are local to
  /// the MPI rank)
  void add(const xtl::span<const T>& x,
           const xtl::span<const std::int32_t>& rows,
           const xtl::span<const std::int32_t>& cols)
  {
    const std::int32_t local_size0 = _index_maps[0]->size_local();
    const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();

    assert(x.size() == rows.size() * cols.size());
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
      {
        throw std::runtime_error("Local row out of range");
      }
    }
  }

  /// Insertion functor to add values to matrix
  /// @param A Matrix to insert into
  static std::function<int(int nr, const int* r, int nc, const int* c,
                           const T* data)>
  mat_add_values(MatrixCSR& A)
  {
    return [&A](int nr, const int* r, int nc, const int* c, const T* data) {
      A.add(tcb::span<const T>(data, nr * nc), tcb::span<const int>(r, nr),
            tcb::span<const int>(c, nc));
      return 0;
    };
  }

  /// Convert to a dense matrix
  /// @param ghost_rows Include ghost rows
  /// @return Dense copy of the matrix
  xt::xtensor<T, 2> to_dense(bool ghost_rows = false) const
  {
    const std::int32_t nrows
        = ghost_rows ? _row_ptr.size() - 1 : _index_maps[0]->size_local();
    const std::int32_t ncols
        = _index_maps[1]->size_local() + _index_maps[1]->num_ghosts();
    xt::xtensor<T, 2> A = xt::zeros<T>({nrows, ncols});
    for (std::size_t r = 0; r < nrows; ++r)
    {
      for (int j = _row_ptr[r]; j < _row_ptr[r + 1]; ++j)
        A(r, _cols[j]) = _data[j];
    }

    return A;
  }

  /// Copy cached ghost values to row owner and add.
  void finalize()
  {
    finalize_begin();
    finalize_end();
  }

  /// Begin transfer of ghost row entries to owning processes
  /// using non-blocking communication with MPI.
  /// Must be followed by finalize_end()
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
        dolfinx::MPI::mpi_type<T>(), _neighbor_comm, &_request);
    assert(status == MPI_SUCCESS);
  }

  /// Complete transfer of ghost rows to owning processes
  /// Must be preceded by finalize_begin()
  void finalize_end()
  {
    int status = MPI_Wait(&_request, MPI_STATUS_IGNORE);
    assert(status == MPI_SUCCESS);

    // Add to local rows
    assert(_ghost_value_data_in.size() == _unpack_pos.size());
    for (std::size_t i = 0; i < _ghost_value_data_in.size(); ++i)
      _data[_unpack_pos[i]] += _ghost_value_data_in[i];

    // Clear cache
    const std::int32_t local_size0 = _index_maps[0]->size_local();
    std::fill(std::next(_data.begin(), _row_ptr[local_size0]), _data.end(), 0);
  }

  /// Index maps for the row and column space. The row IndexMap contains ghost
  /// entries for rows which may be inserted into and the column IndexMap
  /// contains all local and ghost columns that may exist in the owned rows.
  ///
  /// @return Row and column index maps
  const std::array<std::shared_ptr<const common::IndexMap>, 2>&
  index_maps() const
  {
    return _index_maps;
  }

  /// Get local values
  xtl::span<T> values() { return xtl::span(_data); }

  /// Get local row pointers
  xtl::span<std::int32_t> row_ptr() { return xtl::span(_row_ptr); }

  /// Get local column indices
  xtl::span<std::int32_t> cols() { return xtl::span(_cols); }

private:
  // Map describing the data layout for rows and columns
  // including ghost rows and ghost columns
  std::array<std::shared_ptr<const common::IndexMap>, 2> _index_maps;

  // // Block size
  std::array<int, 2> _bs;

  // Data
  std::vector<T, Allocator> _data;
  std::vector<std::int32_t> _cols, _row_ptr;

  // Precomputed data for finalize/update
  // Neighborhood communicator
  MPI_Comm _neighbor_comm;
  // Request in non-blocking communication
  MPI_Request _request;
  // Position in _data to add received data
  std::vector<int> _unpack_pos;
  // Displacements for alltoall for each neighbor when sending and receiving
  std::vector<int> _val_send_disp;
  std::vector<int> _val_recv_disp;
  // Ownership of each row, by neighbor
  std::vector<int> _ghost_row_to_neighbor_rank;
  // Temporary store for finalize data during non-blocking communication
  std::vector<T> _ghost_value_data_in;
};

} // namespace dolfinx::la
