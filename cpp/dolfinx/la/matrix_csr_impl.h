// Copyright (C) 2021-2023 Garth N. Wells and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::la
{
namespace impl
{
/// @brief Incorporate data into a CSR matrix.
///
/// @tparam BS0 Row block size (of both matrix and data).
/// @tparam BS1 Column block size (of both matrix and data).
/// @tparam OP The operation (usually "set" or "add").
/// @param[out] data CSR matrix data.
/// @param[in] cols CSR column indices.
/// @param[in] row_ptr Pointer to the ith row in the CSR data.
/// @param[in] x The `m` by `n` dense block of values (row-major) to add
/// to the matrix.
/// @param[in] xrows Row indices of `x`.
/// @param[in] xcols Column indices of `x`.
/// @param[in] op The operation (set or add),
/// @param[in] num_rows Maximum row index that can be set. Used when
/// debugging to check that rows beyond a permitted range are not being
/// set.
///
/// @note In the case of block data, where BS0 or BS1 are greater than
/// one, the layout of the input data is still the same. For example,
/// the following can be inserted into the top-left corner with
/// `xrows={0,1}` and `xcols={0,1}`, `BS0=2`, `BS1=2` and
/// `x={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}`.
///
/// 0  1  | 2  3
/// 4  5  | 6  7
/// -------------
/// 8  9  | 10 11
/// 12 13 | 14 15
///
template <int BS0, int BS1, typename OP, typename U, typename V, typename W,
          typename X, typename Y>
void insert_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
                const Y& xrows, const Y& xcols, OP op,
                [[maybe_unused]] typename Y::value_type num_rows)
{
  const std::size_t nc = xcols.size();
  assert(x.size() == xrows.size() * xcols.size() * BS0 * BS1);
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    auto row = xrows[r];
    using T = typename X::value_type;
    const T* xr = x.data() + r * nc * BS0 * BS1;

#ifndef NDEBUG
    if (row >= num_rows)
      throw std::runtime_error("Local row out of range");
#endif
    // Columns indices for row
    auto cit0 = std::next(cols.begin(), row_ptr[row]);
    auto cit1 = std::next(cols.begin(), row_ptr[row + 1]);
    for (std::size_t c = 0; c < nc; ++c)
    {
      // Find position of column index
      auto it = std::lower_bound(cit0, cit1, xcols[c]);
      if (it == cit1 or *it != xcols[c])
        throw std::runtime_error("Entry not in sparsity");

      std::size_t d = std::distance(cols.begin(), it);
      std::size_t di = d * BS0 * BS1;
      std::size_t xi = c * BS1;
      assert(di < data.size());
      for (int i = 0; i < BS0; ++i)
      {
        for (int j = 0; j < BS1; ++j)
          op(data[di + j], xr[xi + j]);
        di += BS1;
        xi += nc * BS1;
      }
    }
  }
}

/// @brief Incorporate blocked data with given block sizes into a
/// non-blocked MatrixCSR.
///
/// @note Matrix block size (bs=1). Matrix sparsity must be correct to
/// accept the data.
/// @note See ::insert_csr for data layout.
///
/// @tparam BS0 Row block size of data.
/// @tparam BS1 Column block size of data.
/// @tparam OP The operation (usually "set" or "add").
/// @param[out] data CSR matrix data.
/// @param[in] cols CSR column indices.
/// @param[in] row_ptr Pointer to the ith row in the CSR data.
/// @param[in] x The `m` by `n` dense block of values (row-major) to add
/// to the matrix.
/// @param[in] xrows Row indices of `x`.
/// @param[in] xcols Column indices of `x`.
/// @param[in] op The operation (set or add).
/// @param[in] num_rows Maximum row index that can be set. Used when
/// debugging to check that rows beyond a permitted range are not being
/// set.
template <int BS0, int BS1, typename OP, typename U, typename V, typename W,
          typename X, typename Y>
void insert_blocked_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
                        const Y& xrows, const Y& xcols, OP op,
                        [[maybe_unused]] typename Y::value_type num_rows)
{
  const std::size_t nc = xcols.size();
  assert(x.size() == xrows.size() * xcols.size() * BS0 * BS1);
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    auto row = xrows[r] * BS0;
#ifndef NDEBUG
    if (row >= num_rows)
      throw std::runtime_error("Local row out of range");
#endif

    for (int i = 0; i < BS0; ++i)
    {
      using T = typename X::value_type;
      const T* xr = x.data() + (r * BS0 + i) * nc * BS1;

      // Columns indices for row
      auto cit0 = std::next(cols.begin(), row_ptr[row + i]);
      auto cit1 = std::next(cols.begin(), row_ptr[row + i + 1]);
      for (std::size_t c = 0; c < nc; ++c)
      {
        // Find position of column index
        auto it = std::lower_bound(cit0, cit1, xcols[c] * BS1);
        if (it == cit1 or *it != xcols[c] * BS1)
          throw std::runtime_error("Entry not in sparsity");

        std::size_t d = std::distance(cols.begin(), it);
        assert(d < data.size());
        std::size_t xi = c * BS1;
        for (int j = 0; j < BS1; ++j)
          op(data[d + j], xr[xi + j]);
      }
    }
  }
}

/// @brief Incorporate non-blocked data into a blocked matrix (data
/// block size=1)
///
/// @note Matrix sparsity must be correct to accept the data.
/// @note See ::insert_csr for data layout.
///
/// @param[out] data CSR matrix data.
/// @param[in] cols CSR column indices.
/// @param[in] row_ptr Pointer to the ith row in the CSR data.
/// @param[in] x The `m` by `n` dense block of values (row-major) to add
/// to the matrix.
/// @param[in] xrows Rrow indices of `x`.
/// @param[in] xcols Column indices of `x`.
/// @param[in] op The operation (set or add).
/// @param[in] num_rows Maximum row index that can be set. Used when
/// debugging to check that rows beyond a permitted range are not being
/// set.
/// @param[in] bs0 Row block size of matrix.
/// @param[in] bs1 Column block size of matrix.
template <typename OP, typename U, typename V, typename W, typename X,
          typename Y>
void insert_nonblocked_csr(U&& data, const V& cols, const W& row_ptr,
                           const X& x, const Y& xrows, const Y& xcols, OP op,
                           typename Y::value_type num_rows, int bs0, int bs1)
{
  const std::size_t nc = xcols.size();
  const int nbs = bs0 * bs1;

  assert(x.size() == xrows.size() * xcols.size());
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    auto rdiv = std::div(xrows[r], bs0);
    using T = typename X::value_type;
    const T* xr = x.data() + r * nc;

#ifndef NDEBUG
    if (rdiv.quot >= num_rows)
      throw std::runtime_error("Local row out of range");
#endif
    // Columns indices for row
    auto cit0 = std::next(cols.begin(), row_ptr[rdiv.quot]);
    auto cit1 = std::next(cols.begin(), row_ptr[rdiv.quot + 1]);
    for (std::size_t c = 0; c < nc; ++c)
    {
      // Find position of column index
      auto cdiv = std::div(xcols[c], bs1);
      auto it = std::lower_bound(cit0, cit1, cdiv.quot);
      if (it == cit1 or *it != cdiv.quot)
        throw std::runtime_error("Entry not in sparsity");

      std::size_t d = std::distance(cols.begin(), it);
      std::size_t di = d * nbs + rdiv.rem * bs1 + cdiv.rem;
      assert(di < data.size());
      op(data[di], xr[c]);
    }
  }
}

/// @brief  Sparse matrix-vector product implementation.
/// @tparam T
/// @tparam BS1
/// @param values
/// @param row_begin
/// @param row_end
/// @param indices
/// @param x
/// @param y
/// @param bs0
/// @param bs1
template <typename T, int BS1>
void spmv(std::span<const T> values, std::span<const std::int64_t> row_begin,
          std::span<const std::int64_t> row_end,
          std::span<const std::int32_t> indices, std::span<const T> x,
          std::span<T> y, int bs0, int bs1)
{
  assert(row_begin.size() == row_end.size());
  for (int k0 = 0; k0 < bs0; ++k0)
  {
    for (std::size_t i = 0; i < row_begin.size(); i++)
    {
      T vi{0};
      for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
      {
        if constexpr (BS1 == -1)
        {
          for (int k1 = 0; k1 < bs1; ++k1)
          {
            vi += values[j * bs1 * bs0 + k1 * bs0 + k0]
                  * x[indices[j] * bs1 + k1];
          }
        }
        else
        {
          for (int k1 = 0; k1 < BS1; ++k1)
          {
            vi += values[j * BS1 * bs0 + k1 * bs0 + k0]
                  * x[indices[j] * BS1 + k1];
          }
        }
      }

      y[i * bs0 + k0] += vi;
    }
  }
}

} // namespace impl
} // namespace dolfinx::la
