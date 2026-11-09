// Copyright (C) 2021-2026 Garth N. Wells and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace dolfinx::la
{
namespace impl
{
/// @cond
template <typename T>
struct is_integral_constant : std::false_type
{
};
template <std::integral U, U N>
struct is_integral_constant<std::integral_constant<U, N>> : std::true_type
{
};
/// @endcond

/// Concept for a block size argument: a run-time integral value or a
/// compile-time `std::integral_constant<U, N>` for some integral type
/// `U`.
template <typename T>
concept BlockSizeArg = std::integral<std::remove_cvref_t<T>>
                       or is_integral_constant<std::remove_cvref_t<T>>::value;

/// @brief A block size argument as an `int` or a
/// `std::integral_constant<int, N>`.
/// @param[in] bs Block size of any integral type.
/// @return `bs` with value type `int`, compile-time if `bs` is.
constexpr auto int_block_size(BlockSizeArg auto bs)
{
  using B = std::remove_cvref_t<decltype(bs)>;
  if constexpr (is_integral_constant<B>::value)
    return std::integral_constant<int, B::value>{};
  else
    return static_cast<int>(bs);
}

/// @brief Incorporate data into a CSR matrix.
///
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
/// @param[in] bs0 Row block size of both matrix and data. A run-time
/// integral value or a `std::integral_constant`.
/// @param[in] bs1 Column block size of both matrix and data. A run-time
/// integral value or a `std::integral_constant`.
///
/// @note In the case of block data, where `bs0` or `bs1` are greater
/// than one, the layout of the input data is still the same. For
/// example, the following can be inserted into the top-left corner with
/// `xrows={0,1}` and `xcols={0,1}`, `bs0=2`, `bs1=2` and
/// `x={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}`.
///
/// 0  1  | 2  3
/// 4  5  | 6  7
/// -------------
/// 8  9  | 10 11
/// 12 13 | 14 15
///
template <typename OP, typename U, typename V, typename W, typename X,
          typename Y>
void insert_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
                const Y& xrows, const Y& xcols, OP op,
                [[maybe_unused]] typename Y::value_type num_rows,
                BlockSizeArg auto bs0, BlockSizeArg auto bs1)
{
  const std::size_t nc = xcols.size();
  assert(x.size() == xrows.size() * xcols.size() * bs0 * bs1);
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    auto row = xrows[r];
    using T = typename X::value_type;
    const T* xr = x.data() + r * nc * bs0 * bs1;

#ifndef NDEBUG
    if (row >= num_rows)
      throw std::out_of_range("Local row out of range");
#endif
    // Columns indices for row
    auto cit0 = std::next(cols.begin(), row_ptr[row]);
    auto cit1 = std::next(cols.begin(), row_ptr[row + 1]);
    for (std::size_t c = 0; c < nc; ++c)
    {
      // Find position of column index
      auto it = std::lower_bound(cit0, cit1, xcols[c]);
      if (it == cit1 or *it != xcols[c])
        throw std::out_of_range("Entry not in sparsity");

      std::size_t d = std::ranges::distance(cols.begin(), it);
      std::size_t di = d * bs0 * bs1;
      std::size_t xi = c * bs1;
      assert(di < data.size());
      for (decltype(+bs0) i = 0; i < bs0; ++i)
      {
        for (decltype(+bs1) j = 0; j < bs1; ++j)
          op(data[di + j], xr[xi + j]);
        di += bs1;
        xi += nc * bs1;
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
/// @param[in] bs0 Row block size of data. A run-time integral value or
/// a `std::integral_constant`.
/// @param[in] bs1 Column block size of data. A run-time integral value
/// or a `std::integral_constant`.
template <typename OP, typename U, typename V, typename W, typename X,
          typename Y>
void insert_blocked_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
                        const Y& xrows, const Y& xcols, OP op,
                        [[maybe_unused]] typename Y::value_type num_rows,
                        BlockSizeArg auto bs0, BlockSizeArg auto bs1)
{
  const std::size_t nc = xcols.size();
  assert(x.size() == xrows.size() * xcols.size() * bs0 * bs1);
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    auto row = xrows[r] * bs0;
#ifndef NDEBUG
    if (row + bs0 > num_rows)
      throw std::out_of_range("Local row out of range");
#endif

    for (decltype(+bs0) i = 0; i < bs0; ++i)
    {
      using T = typename X::value_type;
      const T* xr = x.data() + (r * bs0 + i) * nc * bs1;

      // Columns indices for row
      auto cit0 = std::next(cols.begin(), row_ptr[row + i]);
      auto cit1 = std::next(cols.begin(), row_ptr[row + i + 1]);
      for (std::size_t c = 0; c < nc; ++c)
      {
        // Find position of column index
        auto it = std::lower_bound(cit0, cit1, xcols[c] * bs1);
        if (it == cit1 or *it != xcols[c] * bs1)
          throw std::out_of_range("Entry not in sparsity");

        std::size_t d = std::ranges::distance(cols.begin(), it);
        assert(d < data.size());
        std::size_t xi = c * bs1;
        for (decltype(+bs1) j = 0; j < bs1; ++j)
          op(data[d + j], xr[xi + j]);
      }
    }
  }
}

/// @brief Incorporate data into a blocked matrix whose block size
/// differs from the data block size.
///
/// Each matrix block spanned by a data block is located separately,
/// once for all data rows in the same matrix block row, so this is
/// slower than ::insert_csr and ::insert_blocked_csr.
///
/// @note Matrix sparsity must be correct to accept the data.
/// @note See ::insert_csr for data layout.
///
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
/// @param[in] mbs0 Row block size of matrix.
/// @param[in] mbs1 Column block size of matrix.
/// @param[in] bs0 Row block size of data. A run-time integral value or
/// a `std::integral_constant`.
/// @param[in] bs1 Column block size of data. A run-time integral value
/// or a `std::integral_constant`.
template <typename OP, typename U, typename V, typename W, typename X,
          typename Y>
void insert_nonblocked_csr(U&& data, const V& cols, const W& row_ptr,
                           const X& x, const Y& xrows, const Y& xcols, OP op,
                           [[maybe_unused]] typename Y::value_type num_rows,
                           int mbs0, int mbs1, BlockSizeArg auto bs0,
                           BlockSizeArg auto bs1)
{
  // x is row-major, (xrows.size() * bs0) x (xcols.size() * bs1). Entry
  // (i, j) of data block (r, c) is matrix entry
  // (xrows[r] * bs0 + i, xcols[c] * bs1 + j), which lies in matrix
  // block (p, q) of size mbs0 x mbs1.
  const std::size_t nc = xcols.size();
  const std::size_t x_row_size = nc * bs1;
  const int nbs = mbs0 * mbs1;
  assert(x.size() == xrows.size() * x_row_size * bs0);
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Looping over all bs0 rows, rather than stepping by ni, keeps a
    // compile-time bound
    for (decltype(+bs0) i = 0; i < bs0; ++i)
    {
      // Map data row i to row block_row of matrix block row p
      const std::int32_t row = xrows[r] * bs0 + i;
      const std::int32_t p = row / mbs0;
      const std::int32_t block_row = row % mbs0;

      // Already written with the first data row in p
      if (i > 0 and block_row > 0)
        continue;

      // Data rows i, ..., i + ni - 1 lie in p and are written together
      const int ni = std::min<int>(bs0 - i, mbs0 - block_row);
      const typename X::value_type* xr = x.data() + (r * bs0 + i) * x_row_size;

#ifndef NDEBUG
      if (p >= num_rows)
        throw std::out_of_range("Local row out of range");
#endif
      // Locate block column q in row p, searching from first.
      // std::lower_bound rather than std::ranges::lower_bound: GCC lays
      // out the latter's search loop with a taken branch on every step
      // right, which is measurably slower for sub-space insertion.
      auto cit0 = std::next(cols.begin(), row_ptr[p]);
      auto cit1 = std::next(cols.begin(), row_ptr[p + 1]);
      auto locate = [&cit1](auto first, std::int32_t q)
      {
        auto it = std::lower_bound(first, cit1, q);
        if (it == cit1 or *it != q)
          throw std::out_of_range("Entry not in sparsity");
        return it;
      };

      for (std::size_t c = 0; c < nc; ++c)
      {
        // Map data column j to column block_col of matrix block column q
        const std::int32_t col = xcols[c] * bs1;
        std::int32_t q = col / mbs1;
        std::int32_t block_col = col % mbs1;
        auto it = locate(cit0, q);
        for (decltype(+bs1) j = 0; j < bs1; ++j, ++block_col)
        {
          // Crossed into block column q + 1, which follows q in cols
          if (block_col == mbs1)
          {
            it = locate(it, ++q);
            block_col = 0;
          }

          // Entry (i + k, j) of the data block is entry
          // (block_row + k, block_col) of matrix block (p, q), the d-th
          // block in data. Carrying d (or d * nbs) across j instead
          // costs GCC an extra instruction per data column, which is
          // measurably slower for sub-space insertion.
          const std::size_t d = std::ranges::distance(cols.begin(), it);
          std::size_t di = d * nbs + block_row * mbs1 + block_col;
          std::size_t xi = c * bs1 + j;
          for (int k = 0; k < ni; ++k, di += mbs1, xi += x_row_size)
          {
            assert(di < data.size());
            op(data[di], xr[xi]);
          }
        }
      }
    }
  }
}

/// @brief Sparse matrix-vector product implementation.
///
/// Computes `y += A x` where A is given in CSR format.
///
/// @note The value block layout is row-major within each block: for
/// block entry `j` the element at row-offset `k0` and column-offset `k1`
/// is stored at `values[j * bs0 * bs1 + k0 * bs1 + k1]`.
///
/// @tparam T Scalar type of the matrix and vector entries.
/// @param[in] values Nonzero values of A, stored block-row-major.
/// Length: `nnz * bs0 * bs1`.
/// @param[in] row_begin Start positions in `values`/`indices` for each
/// row of A. Length: number of rows of A.
/// @param[in] row_end End positions in `values`/`indices` for each row
/// of A. Length: number of rows of A.
/// @param[in] indices Column indices of each nonzero block entry of A.
/// Length: `nnz`.
/// @param[in] x Input vector, indexed by the *columns* of A.
/// Length: `num_cols * bs1`.
/// @param[in,out] y Output vector, indexed by the *rows* of A,
/// accumulated into. Length: `num_rows * bs0`.
/// @param[in] bs0 Row block size. Either a runtime integral value or a
/// `std::integral_constant<U, N>` for some integral type `U`.
/// @param[in] bs1 Column block size. Either a runtime integral value
/// or a `std::integral_constant<U, N>` for some integral type `U`.
template <typename T>
void spmv(std::span<const T> values, std::span<const std::int64_t> row_begin,
          std::span<const std::int64_t> row_end,
          std::span<const std::int32_t> indices, std::span<const T> x,
          std::span<T> y, BlockSizeArg auto bs0, BlockSizeArg auto bs1)
{
  assert(row_begin.size() == row_end.size());
  // Block layout: row-major within each block. The element at row-offset
  // k0 (∈ [0, bs0)) and column-offset k1 (∈ [0, bs1)) of block entry j
  // is stored at values[j * bs0 * bs1 + k0 * bs1 + k1].
  for (decltype(+bs0) k0 = 0; k0 < bs0; ++k0)
  {
    for (std::size_t i = 0; i < row_begin.size(); i++)
    {
      T vi{0};
      for (std::int64_t j = row_begin[i]; j < row_end[i]; j++)
      {
        for (decltype(+bs1) k1 = 0; k1 < bs1; ++k1)
        {
          vi += values[j * bs0 * bs1 + k0 * bs1 + k1]
                * x[indices[j] * bs1 + k1];
        }
      }

      y[i * bs0 + k0] += vi;
    }
  }
}

/// @brief Sparse matrix-vector transpose product implementation.
///
/// Computes `y += A^T x` where A is given in CSR format.  The transpose
/// is applied implicitly: for each nonzero A(i, indices[j]) the
/// contribution `A(i,j) * x[i]` is scattered into `y[indices[j]]`.
///
/// @note `y` is accumulated into (not overwritten).  Callers should
/// zero `y` before the first call if a fresh result is required.
///
/// @note The value block layout is row-major within each block: for
/// block entry `j` the element at row-offset `k0` and column-offset
/// `k1` is stored at `values[j * bs0 * bs1 + k0 * bs1 + k1]`.
///
/// @tparam T Scalar type of the matrix and vector entries.
///
/// @param[in] values Nonzero values of A, stored block-row-major.
/// Length: `nnz * bs0 * bs1`.
/// @param[in] row_begin Start positions in `values`/`indices` for each
/// row of A.  Length: number of rows of A.
/// @param[in] row_end End positions in `values`/`indices` for each row
/// of A.  Length: number of rows of A.
/// @param[in] indices Column indices of each nonzero block entry of A.
/// Length: `nnz`.
/// @param[in] x Input vector, indexed by the *rows* of A. Length:
/// `num_rows * bs0`.
/// @param[in,out] y Output vector, indexed by the *columns* of A,
/// accumulated into. Length: `num_cols * bs1`.
/// @param[in] bs0 Row block size. Either a runtime integral value or a
/// `std::integral_constant<U, N>` for some integral type `U`.
/// @param[in] bs1 Column block size. Either a runtime integral value
/// or a `std::integral_constant<U, N>` for some integral type `U`.
template <typename T>
void spmvT(std::span<const T> values, std::span<const std::int64_t> row_begin,
           std::span<const std::int64_t> row_end,
           std::span<const std::int32_t> indices, std::span<const T> x,
           std::span<T> y, BlockSizeArg auto bs0, BlockSizeArg auto bs1)
{
  assert(row_begin.size() == row_end.size());

  // Block layout: row-major within each block.
  for (decltype(+bs0) k0 = 0; k0 < bs0; ++k0)
  {
    for (std::size_t i = 0; i < row_begin.size(); i++)
    {
      const T xval = x[i * bs0 + k0];
      for (std::int64_t j = row_begin[i]; j < row_end[i]; j++)
      {
        for (decltype(+bs1) k1 = 0; k1 < bs1; ++k1)
        {
          y[indices[j] * bs1 + k1]
              += values[j * bs0 * bs1 + k0 * bs1 + k1] * xval;
        }
      }
    }
  }
}

} // namespace impl
} // namespace dolfinx::la
