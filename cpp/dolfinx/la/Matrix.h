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
        _ncols(_index_maps[1]->size_local() + _index_maps[1]->num_ghosts())
  {
    // TODO: handle block sizes
    // TODO: check that column indices for each row in p are sorted
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
    assert(x.size() == rows.size() * cols.size());
    for (std::size_t r = 0; r < rows.size(); ++r)
    {
      // Columns indices for row
      std::int32_t row = rows[r];
      auto cit0 = std::next(_cols.begin(), _row_ptr[row]);
      auto cit1 = std::next(_cols.begin(), _row_ptr[row + 1]);

      // Current data row
      const T* xr = x.data() + r * cols.size();

      for (std::size_t c = 0; c < cols.size(); ++c)
      {
        // Find position of column index
        auto it = std::find(cit0, cit1, cols[c]);
        assert(it != cit1);
        std::size_t d = std::distance(_cols.begin(), it);
        _data[d] = op(_data[d], xr[c]);
      }
    }
  }

  /// Convert to a dense matrix
  /// @return Dense copy of the matrix
  xt::xtensor<T, 2> to_dense() const
  {
    std::int32_t nrows = _row_ptr.size() - 1;
    xt::xtensor<T, 2> A = xt::zeros<T>({nrows, _ncols});
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

private:
  // Map describing the data layout for rows and columns
  std::vector<std::shared_ptr<const common::IndexMap>> _index_maps;

  // // Block size
  // int _bs;

  // Data
  std::vector<T, Allocator> _data;
  std::vector<std::int32_t> _cols, _row_ptr;

  std::int32_t _ncols = -1;
};

} // namespace dolfinx::la
