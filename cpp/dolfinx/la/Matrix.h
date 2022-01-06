// Copyright (C) 2021 Garth N. Wells
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
        _row_ptr(_index_maps[0]->size_local() + 1, 0),
        _index_cache(_index_maps[0]->num_ghosts()),
        _value_cache(_index_maps[0]->num_ghosts())
  {
    // TODO: handle block sizes
    // TODO: support distributed matrices

    const graph::AdjacencyList<std::int32_t>& pg = p.graph();
    std::copy(pg.array().begin(), pg.array().end(), _cols.begin());
    std::copy(pg.offsets().begin(), pg.offsets().end(), _row_ptr.begin());
  }

  /// Set all non-zero entries to a value
  void set(T x) { std::fill(_data.begin(), _data.end(), x); }

  /// Add
  /// @param[in] x The `m` by `n` dense block of values (row-major) to
  /// add to the matrix
  /// @param[in] rows The row indices of `x` (indices are local to the MPI rank)
  /// @param[in] cols The column indices of `x` (indices are local to
  /// the MPI rank)
  /// @param[in] op
  void add(const xtl::span<const T>& x,
           const xtl::span<const std::int32_t>& rows,
           const xtl::span<const std::int32_t>& cols,
           std::function<T(T, T)> op = std::plus<T>())
  {
    const std::int32_t local_size0 = _index_maps[0]->size_local();
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
          auto it = std::find(cit0, cit1, cols[c]);
          assert(it != cit1);
          std::size_t d = std::distance(_cols.begin(), it);
          _data[d] = op(_data[d], xr[c]);
        }
      }
      else
      {
        for (std::size_t c = 0; c < cols.size(); ++c)
        {
          _index_cache[row - local_size0].push_back(cols[c]);
          _value_cache[row - local_size0].push_back(xr[c]);
        }
      }
    }
  }

  /// Insertion functor with a general operation
  /// @param A Matrix to insert into
  /// @param op Operation (add by default)
  static std::function<int(int nr, const int* r, int nc, const int* c,
                           const T* data)>
  mat_insert_values(Matrix& A, std::function<T(T, T)> op = std::plus<T>())
  {
    return
        [&A, &op](int nr, const int* r, int nc, const int* c, const T* data) {
          A.add(tcb::span<const T>(data, nr * nc), tcb::span<const int>(r, nr),
                tcb::span<const int>(c, nc), op);
          return 0;
        };
  }

  /// Convert to a dense matrix
  /// @return Dense copy of the matrix
  xt::xtensor<T, 2> to_dense() const
  {
    std::int32_t nrows = _row_ptr.size() - 1;
    std::int32_t ncols
        = _index_maps[1]->size_local() + _index_maps[1]->num_ghosts();
    xt::xtensor<T, 2> A = xt::zeros<T>({nrows, ncols});
    for (std::size_t r = 0; r < nrows; ++r)
    {
      auto cit0 = std::next(_cols.begin(), _row_ptr[r]);
      auto cit1 = std::next(_cols.begin(), _row_ptr[r + 1]);
      for (auto it = cit0; it != cit1; ++it)
      {
        std::size_t pos = std::distance(_cols.begin(), it);
        A(r, *it) = _data[pos];
      }
    }

    return A;
  }

  /// Copy cached ghost values to row owner.
  void finalize(std::function<T(T, T)> op = std::plus<T>())
  {
    const std::int32_t local_size0 = _index_maps[0]->size_local();
    const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();
    const std::array local_range0 = _index_maps[0]->local_range();
    const std::vector<std::int64_t>& ghosts0 = _index_maps[0]->ghosts();
    const std::vector<int> ghost_owners0 = _index_maps[0]->ghost_owner_rank();

    const std::int32_t local_size1 = _index_maps[1]->size_local();
    const std::array local_range1 = _index_maps[1]->local_range();
    const std::vector<std::int64_t>& ghosts1 = _index_maps[1]->ghosts();

    // Get ghost->owner communicator for rows
    MPI_Comm comm = _index_maps[0]->comm(common::IndexMap::Direction::reverse);
    const auto dest_ranks = dolfinx::MPI::neighbors(comm)[1];

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
      data_per_proc[ghost_to_neighbour_rank[i]] += _value_cache[i].size();
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
      const std::vector<std::int32_t>& col_cache_i = _index_cache[i];
      const std::vector<T>& val_cache_i = _value_cache[i];

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
        = MPI::neighbor_all_to_all(comm, ghost_index_data_out).array();

    const graph::AdjacencyList<T> ghost_value_data_out(
        std::move(ghost_value_data), std::move(val_send_disp));
    const std::vector<T> ghost_value_data_in
        = MPI::neighbor_all_to_all(comm, ghost_value_data_out).array();

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
      auto cit = std::find(cit0, cit1, local_col);
      assert(cit != cit1);
      std::size_t d = std::distance(_cols.begin(), cit);
      _data[d] = op(_data[d], ghost_value_data_in[i]);
    }

    // Clear cache
    std::vector<std::vector<std::int32_t>>(num_ghosts0).swap(_index_cache);
    std::vector<std::vector<T>>(num_ghosts0).swap(_value_cache);
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

private:
  // Map describing the data layout for rows and columns
  std::array<std::shared_ptr<const common::IndexMap>, 2> _index_maps;

  // // Block size
  // int _bs;

  // Data
  std::vector<T, Allocator> _data;
  std::vector<std::int32_t> _cols, _row_ptr;

  // Caching for off-process rows
  std::vector<std::vector<std::int32_t>> _index_cache;
  std::vector<std::vector<T>> _value_cache;
};

} // namespace dolfinx::la
