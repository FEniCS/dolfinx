// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "SparsityPattern.h"
// #include <complex>
// #include <dolfinx/common/IndexMap.h>
// #include <limits>
// #include <memory>
// #include <numeric>
#include <dolfinx/graph/AdjacencyList.h>
#include <vector>
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
      : _data(p.num_nonzeros(), 0, alloc), _cols(p.num_nonzeros()),
        _row_ptr(p.index_map(0)->size_local() + p.index_map(0)->num_ghosts())
  {
    // TODO: handle block sizes
    // TODO: check that column indices for each row in p are sorted
    // TODO: support distributed matrices

    const graph::AdjacencyList<std::int32_t>& dp = p.diagonal_pattern();
    std::copy(dp.array().begin(), dp.array().end(), _cols.begin());
    std::copy(dp.offsets().begin(), dp.offsets().end(), _row_ptr.begin());
  }

  /// Add
  void add(const xtl::span<const T>& x,
           const xtl::span<const std::int32_t>& rows,
           const xtl::span<const std::int32_t>& cols)
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
        _data[d] += xr[c];
      }
    }
  }

  // void set(const xtl::span<const T>& x,
  //          const xtl::span<const std::int32_t>& rows,
  //          const xtl::span<const std::int32_t>& cols)
  // {
  //   assert(x.size() == rows.size() * cols.size());
  //   for (std::size_t r = 0; r < rows.size(); ++r)
  //   {
  //     // Columns indices for row
  //     auto cit0 = std::next(_cols.begin(), _row_ptr[row]);
  //     auto cit1 = std::next(_cols.begin(), _row_ptr[row + 1]);

  //     // Current data row
  //     const T* xr = x.data() + r * cols.size();

  //     for (std::size_t c = 0; c < cols.size(); ++c)
  //     {
  //       // Find position of column index
  //       auto it = std::find(cit0, cit1, cols[c]);
  //       assert(it != cit1);
  //       std::size_t d = std::distance(_cols.begin(), it);
  //       _data[d] = xr[c];
  //     }
  //   }
  // }

private:
  // // Map describing the data layout
  // std::shared_ptr<const common::IndexMap> _map;

  // // Block size
  // int _bs;

  // // Data type and buffers for ghost scatters
  // MPI_Datatype _datatype = MPI_DATATYPE_NULL;
  // MPI_Request _request = MPI_REQUEST_NULL;
  // std::vector<T> _buffer_send_fwd, _buffer_recv_fwd;

  // Data
  std::vector<T, Allocator> _data;
  std::vector<std::int32_t> _cols, _row_ptr;
};

} // namespace dolfinx::la
