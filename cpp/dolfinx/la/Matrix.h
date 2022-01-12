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

/// Distributed sparse Matrix
template <typename T, class Allocator = std::allocator<T>>
class Matrix
{
public:
  /// The value type
  using value_type = T;

  /// Create a distributed matrix
  Matrix(const SparsityPattern& p, const Allocator& alloc = Allocator())
      : _index_maps({p.index_map(0), p.column_index_map()}),
        _data(p.num_nonzeros(), 0, alloc), _cols(p.num_nonzeros()),
        _row_ptr(
            _index_maps[0]->size_local() + _index_maps[0]->num_ghosts() + 1, 0)
  {
    // TODO: handle block sizes

    // Comm for ghost updates
    _nbr_comm = _index_maps[0]->comm(common::IndexMap::Direction::reverse);

    const graph::AdjacencyList<std::int32_t>& pg = p.graph();
    std::copy(pg.array().begin(), pg.array().end(), _cols.begin());
    std::copy(pg.offsets().begin(), pg.offsets().end(), _row_ptr.begin());
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
  mat_set_values(Matrix& A)
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
  mat_add_values(Matrix& A)
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
    // TODO: move some of this to the constructor and/or share data from
    // SparsityPattern

    const std::int32_t local_size0 = _index_maps[0]->size_local();
    const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();
    const std::array local_range0 = _index_maps[0]->local_range();
    const std::vector<std::int64_t>& ghosts0 = _index_maps[0]->ghosts();
    const std::vector<int> ghost_owners0 = _index_maps[0]->ghost_owner_rank();

    const std::int32_t local_size1 = _index_maps[1]->size_local();
    const std::array local_range1 = _index_maps[1]->local_range();
    const std::vector<std::int64_t>& ghosts1 = _index_maps[1]->ghosts();

    // Get ghost->owner communicator for rows
    const auto dest_ranks = dolfinx::MPI::neighbors(_nbr_comm)[1];

    // Global-to-neigbourhood map for destination ranks
    std::map<int, std::int32_t> dest_proc_to_neighbor;
    for (std::size_t i = 0; i < dest_ranks.size(); ++i)
      dest_proc_to_neighbor.insert({dest_ranks[i], i});

    // Compute size of data to send to each process
    std::vector<std::int32_t> data_per_proc(dest_ranks.size(), 0);
    std::vector<int> ghost_to_neighbour_rank(num_ghosts0, -1);
    for (int i = 0; i < num_ghosts0; ++i)
    {
      // Find rank on neigbourhood comm of ghost owner
      const auto it = dest_proc_to_neighbor.find(ghost_owners0[i]);
      assert(it != dest_proc_to_neighbor.end());
      ghost_to_neighbour_rank[i] = it->second;

      // Add to src size
      assert(ghost_to_neighbour_rank[i] < (int)data_per_proc.size());
      data_per_proc[ghost_to_neighbour_rank[i]]
          += _row_ptr[local_size0 + i + 1] - _row_ptr[local_size0 + i];
    }

    // Compute send displacements for values and indices (x2)
    std::vector<int> val_send_disp(dest_ranks.size() + 1, 0);
    std::partial_sum(data_per_proc.begin(), data_per_proc.end(),
                     std::next(val_send_disp.begin(), 1));
    std::vector<int> index_send_disp(dest_ranks.size() + 1);
    std::transform(val_send_disp.begin(), val_send_disp.end(),
                   index_send_disp.begin(), [](int d) { return d * 2; });

    // For each ghost row, pack and send values to send to neighborhood
    std::vector<int> insert_pos(val_send_disp);
    std::vector<std::int64_t> ghost_index_data(index_send_disp.back());
    std::vector<T> ghost_value_data(val_send_disp.back());
    for (int i = 0; i < num_ghosts0; ++i)
    {
      const int neighbour_rank = ghost_to_neighbour_rank[i];
      const xtl::span<std::int32_t> col_cache_i(
          _cols.data() + _row_ptr[local_size0 + i],
          _row_ptr[local_size0 + i + 1] - _row_ptr[local_size0 + i]);
      const xtl::span<T> val_cache_i(_data.data() + _row_ptr[local_size0 + i],
                                     _row_ptr[local_size0 + i + 1]
                                         - _row_ptr[local_size0 + i]);
      for (std::size_t j = 0; j < col_cache_i.size(); ++j)
      {
        std::int32_t col_local = col_cache_i[j];
        // Get index in send buffer
        const std::int32_t idx_pos = 2 * insert_pos[neighbour_rank];
        const std::int32_t val_pos = insert_pos[neighbour_rank];

        // Pack send data (row, col) as global indices
        ghost_index_data[idx_pos] = ghosts0[i];
        if (col_local < local_size1)
          ghost_index_data[idx_pos + 1] = col_local + local_range1[0];
        else
          ghost_index_data[idx_pos + 1] = ghosts1[col_local - local_size1];

        // Send value
        ghost_value_data[val_pos] = val_cache_i[j];

        insert_pos[neighbour_rank]++;
      }
    }

    // Create and communicate adjacency list to neighborhood
    const graph::AdjacencyList<std::int64_t> ghost_index_data_out(
        std::move(ghost_index_data), std::move(index_send_disp));
    const std::vector<std::int64_t> ghost_index_data_in
        = MPI::neighbor_all_to_all(_nbr_comm, ghost_index_data_out).array();

    const graph::AdjacencyList<T> ghost_value_data_out(
        std::move(ghost_value_data), std::move(val_send_disp));
    const std::vector<T> ghost_value_data_in
        = MPI::neighbor_all_to_all(_nbr_comm, ghost_value_data_out).array();

    // Global to local map for ghost columns
    std::map<std::int64_t, std::int32_t> global_to_local;
    std::int32_t local_i = local_size1;
    for (std::int64_t global_i : ghosts1)
      global_to_local.insert({global_i, local_i++});

    assert(ghost_index_data_in.size() == 2 * ghost_value_data_in.size());

    for (std::size_t i = 0; i < ghost_value_data_in.size(); ++i)
    {
      // Row must be on this process
      const std::int32_t local_row
          = ghost_index_data_in[2 * i] - local_range0[0];
      assert(local_row >= 0 and local_row < local_size0);

      // Column may be owned or unowned
      std::int32_t local_col = ghost_index_data_in[2 * i + 1] - local_range1[0];
      if (local_col < 0 or local_col >= local_size1)
      {
        const auto it = global_to_local.find(ghost_index_data_in[2 * i + 1]);
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
      _data[d] += ghost_value_data_in[i];
    }

    // Clear cache
    std::fill(std::next(_data.begin(), _row_ptr[local_size0]), _data.end(), 0);
  }

  /// More efficient update routine, using precomputed data
  void final2()
  {
    // Pack ghost rows to owners
    std::vector<int> insert_pos(_val_send_disp);
    std::vector<T> ghost_value_data(_val_send_disp.back());
    for (int i = 0; i < num_ghosts0; ++i)
    {
      const int neighbour_rank = _ghost_row_to_neighbour_rank[i];
      // Get position in send buffer
      const std::int32_t val_pos = insert_pos[neighbour_rank];
      std::copy(std::next(_data.data(), _row_ptr[local_size0 + i]),
                std::next(_data.data(), _row_ptr[local_size0 + i + 1]),
                std::next(ghost_value_data.begin(), val_pos));

      insert_pos[neighbour_rank]
          += _row_ptr[local_size0 + i + 1] - _row_ptr[local_size0 + i];
    }

    const graph::AdjacencyList<T> ghost_value_data_out(
        std::move(ghost_value_data), std::move(_val_send_disp));
    const std::vector<T> ghost_value_data_in
        = MPI::neighbor_all_to_all(_nbr_comm, ghost_value_data_out).array();

    assert(ghost_value_data_in.size() == _unpack_pos.size());

    // Unpack ghost values to columns
    for (std::size_t i = 0; i < ghost_value_data_in.size(); ++i)
      _data[_unpack_pos[i]] += ghost_value_data_in[i];
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
  // int _bs;

  // Data
  std::vector<T, Allocator> _data;
  std::vector<std::int32_t> _cols, _row_ptr;

  // Precomputed data for finalize/update
  MPI_Comm _nbr_comm;
  std::vector<int> _unpack_pos;
  std::vector<int> _val_send_disp;
};

} // namespace dolfinx::la
