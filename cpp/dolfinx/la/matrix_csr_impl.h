// Copyright (C) 2021-2023 Garth N. Wells and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <iostream>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::la
{
namespace impl
{
/// @brief Incorporate data into a CSR matrix
///
/// @tparam BS0 Row block size (of both matrix and data)
/// @tparam BS1 Column block size (of both matrix and data)
/// @tparam OP The operation (usually "set" or "add")
/// @param[out] data The CSR matrix data
/// @param[in] cols The CSR column indices
/// @param[in] row_ptr The pointer to the ith row in the CSR data
/// @param[in] x The `m` by `n` dense block of values (row-major) to add
/// to the matrix
/// @param[in] xrows The row indices of `x`
/// @param[in] xcols The column indices of `x`
/// @param[in] op The operation (set or add)
/// @param[in] local_size The maximum row index that can be set. Used
/// when debugging is own to check that rows beyond a permitted range
/// are not being set.
///
/// @note In the case of block data, where BS0 or BS1 are greater than
/// one, the layout of the input data is still the same. For example, the
/// following can be inserted into the top-left corner
/// with xrows={0,1} and xcols={0,1}, BS0=2, BS1=2 and
/// x={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}.
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
                typename Y::value_type local_size);

/// @brief Incorporate blocked data with given block sizes into a non-blocked
/// MatrixCSR
/// @note Matrix block size (bs=1). Matrix sparsity must be correct to accept
/// the data.
/// @note see `insert_csr` for data layout
///
/// @tparam BS0 Row block size of Data
/// @tparam BS1 Column block size of Data
/// @tparam OP The operation (usually "set" or "add")
/// @param[out] data The CSR matrix data
/// @param[in] cols The CSR column indices
/// @param[in] row_ptr The pointer to the ith row in the CSR data
/// @param[in] x The `m` by `n` dense block of values (row-major) to add
/// to the matrix
/// @param[in] xrows The row indices of `x`
/// @param[in] xcols The column indices of `x`
/// @param[in] op The operation (set or add)
/// @param[in] local_size The maximum row index that can be set. Used
/// when debugging is own to check that rows beyond a permitted range
/// are not being set.
template <int BS0, int BS1, typename OP, typename U, typename V, typename W,
          typename X, typename Y>
void insert_blocked_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
                        const Y& xrows, const Y& xcols, OP op,
                        typename Y::value_type local_size);

/// @brief Incorporate non-blocked data into a blocked matrix (Data block size =
/// 1)
/// @note Matrix sparsity must be correct to accept the data
/// @note see `insert_csr` for data layout
/// @param[out] data The CSR matrix data
/// @param[in] cols The CSR column indices
/// @param[in] row_ptr The pointer to the ith row in the CSR data
/// @param[in] x The `m` by `n` dense block of values (row-major) to add
/// to the matrix
/// @param[in] xrows The row indices of `x`
/// @param[in] xcols The column indices of `x`
/// @param[in] op The operation (set or add)
/// @param[in] local_size The maximum row index that can be set. Used
/// when debugging is own to check that rows beyond a permitted range
/// are not being set.
/// @param[in] bs0 Row block size of Matrix
/// @param[in] bs1 Column block size of Matrix
template <typename OP, typename U, typename V, typename W, typename X,
          typename Y>
void insert_nonblocked_csr(U&& data, const V& cols, const W& row_ptr,
                           const X& x, const Y& xrows, const Y& xcols, OP op,
                           typename Y::value_type local_size, int bs0, int bs1);

} // namespace impl

//-----------------------------------------------------------------------------
template <int BS0, int BS1, typename OP, typename U, typename V, typename W,
          typename X, typename Y>
void impl::insert_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
                      const Y& xrows, const Y& xcols, OP op,
                      [[maybe_unused]] typename Y::value_type local_size)
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
    if (row >= local_size)
      throw std::runtime_error("Local row out of range");
#endif
    // Columns indices for row
    auto cit0 = std::next(cols.begin(), row_ptr[row]);
    auto cit1 = std::next(cols.begin(), row_ptr[row + 1]);
    for (std::size_t c = 0; c < nc; ++c)
    {
      // Find position of column index
      auto it = std::lower_bound(cit0, cit1, xcols[c]);
      assert(*it == xcols[c]);
      assert(it != cit1);

      std::size_t d = std::distance(cols.begin(), it);
      int di = d * BS0 * BS1;
      int xi = c * BS1;
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
//-----------------------------------------------------------------------------
// Insert with block insertion into a regular CSR (block size 1)
template <int BS0, int BS1, typename OP, typename U, typename V, typename W,
          typename X, typename Y>
void impl::insert_blocked_csr(U&& data, const V& cols, const W& row_ptr,
                              const X& x, const Y& xrows, const Y& xcols, OP op,
                              [[maybe_unused]]
                              typename Y::value_type local_size)
{
  const std::size_t nc = xcols.size();
  assert(x.size() == xrows.size() * xcols.size() * BS0 * BS1);
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    auto row = xrows[r] * BS0;

#ifndef NDEBUG
    if (row >= local_size)
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
        assert(*it == xcols[c] * BS1);
        assert(it != cit1);
        std::size_t d = std::distance(cols.begin(), it);
        assert(d < data.size());
        int xi = c * BS1;
        for (int j = 0; j < BS1; ++j)
          op(data[d + j], xr[xi + j]);
      }
    }
  }
}
//-----------------------------------------------------------------------------
// Add individual entries in block-CSR storage
template <typename OP, typename U, typename V, typename W, typename X,
          typename Y>
void impl::insert_nonblocked_csr(U&& data, const V& cols, const W& row_ptr,
                                 const X& x, const Y& xrows, const Y& xcols,
                                 OP op,
                                 [[maybe_unused]]
                                 typename Y::value_type local_size,
                                 int bs0, int bs1)
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
    if (rdiv.quot >= local_size)
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
      assert(it != cit1);
      assert(*it == cdiv.quot);

      std::size_t d = std::distance(cols.begin(), it);
      const int di = d * nbs + rdiv.rem * bs1 + cdiv.rem;
      assert(di < data.size());
      op(data[di], xr[c]);
    }
  }
}
//-----------------------------------------------------------------------------
} // namespace dolfinx::la
