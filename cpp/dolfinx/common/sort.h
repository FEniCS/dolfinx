// Copyright (C) 2021-2025 Igor Baratta and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace dolfinx
{
struct __unsigned_projection
{
  // Transforms the projected value to an unsigned int (if signed),
  // while maintaining relative order by
  //    x ↦ x + |std::numeric_limits<I>::min()|
  template <std::signed_integral T>
  constexpr std::make_unsigned_t<T> operator()(T e) const noexcept
  {
    using uT = std::make_unsigned_t<T>;

    // Assert binary structure for bit shift
    static_assert(static_cast<uT>(std::numeric_limits<T>::min())
                      + static_cast<uT>(std::numeric_limits<T>::max())
                  == static_cast<uT>(T(-1)));
    static_assert(std::numeric_limits<uT>::digits
                  == std::numeric_limits<T>::digits + 1);
    static_assert(std::bit_cast<uT>(std::numeric_limits<T>::min())
                  == (uT(1) << (sizeof(T) * 8 - 1)));

    return std::bit_cast<uT>(std::forward<T>(e))
           ^ (uT(1) << (sizeof(T) * 8 - 1));
  }
};

/// Projection from signed to signed int
inline constexpr __unsigned_projection unsigned_projection{};

/// @brief Sort a range with radix sorting algorithm. The bucket size is
/// determined by the number of bits to sort at a time (2^BITS).
///
/// This allows usage with standard range containers of integral types,
/// for example
/// @code
/// std::array<std::int16_t, 3> a{2, 3, 1};
/// dolfinx::radix_sort(a); // a = {1, 2, 3}
/// @endcode
/// Additionally the projection based approach of the STL library is
/// adapted, which allows for versatile usage, for example the easy
/// realization of an argsort
/// @code
/// std::array<std::int16_t, 3> a{2, 3, 1};
/// std::array<std::int16_t, 3> i{0, 1, 2};
/// dolfinx::radix_sort(i, [&](auto i){ return a[i]; }); // yields i = {2, 0,
/// 1} and a[i] = {1, 2, 3};
/// @endcode
/// @tparam BITS The number of bits to sort at a time.
/// @tparam P Projection type to be applied on range elements to produce
/// a sorting index.
/// @tparam R Type of the range to sort.
/// @param[in, out] range The range to sort.
/// @param[in] proj Element projection.
template <int BITS = 8, typename P = std::identity,
          std::ranges::random_access_range R>
constexpr void radix_sort(R&& range, P proj = {})
{
  using bits_t = std::make_unsigned_t<
      std::remove_cvref_t<std::invoke_result_t<P, std::iter_value_t<R>>>>;
  constexpr bits_t _BITS = BITS;

  // Value type
  using T = std::iter_value_t<R>;

  // Index type (if no projection is provided it holds I == T)
  using I = std::remove_cvref_t<std::invoke_result_t<P, T>>;
  using uI = std::make_unsigned_t<I>;

  if constexpr (!std::is_same_v<uI, I>)
  {
    radix_sort<_BITS>(std::forward<R>(range), [&](const T& e) -> uI
                      { return unsigned_projection(proj(e)); });
    return;
  }

  if (range.size() <= 1)
    return;

  uI max_value = proj(*std::ranges::max_element(range, std::less{}, proj));

  // Sort N bits at a time
  constexpr uI bucket_size = 1 << _BITS;
  uI mask = (uI(1) << _BITS) - 1;

  // Compute number of iterations, most significant digit (N bits) of
  // maxvalue
  I its = 0;

  // Optimize for case where all first bits are set - then order will
  // not depend on it
  if (bool all_first_bit = std::ranges::all_of(
          range, [&proj](const auto& e)
          { return proj(e) & (uI(1) << (sizeof(uI) * 8 - 1)); });
      all_first_bit)
  {
    max_value = max_value & ~(uI(1) << (sizeof(uI) * 8 - 1));
  }

  while (max_value)
  {
    max_value >>= _BITS;
    its++;
  }

  // Adjacency list arrays for computing insertion position
  std::array<I, bucket_size> counter;
  std::array<I, bucket_size + 1> offset;

  uI mask_offset = 0;
  std::vector<T> buffer(range.size());
  std::span<T> current_perm = range;
  std::span<T> next_perm = buffer;
  for (I i = 0; i < its; i++)
  {
    // Zero counter array
    std::ranges::fill(counter, 0);

    // Count number of elements per bucket
    for (auto c : current_perm)
      counter[(proj(c) & mask) >> mask_offset]++;

    // Prefix sum to get the inserting position
    offset[0] = 0;
    std::partial_sum(counter.begin(), counter.end(), std::next(offset.begin()));
    for (auto c : current_perm)
    {
      uI bucket = (proj(c) & mask) >> mask_offset;
      uI new_pos = offset[bucket + 1] - counter[bucket];
      next_perm[new_pos] = c;
      counter[bucket]--;
    }

    mask = mask << _BITS;
    mask_offset += _BITS;

    std::swap(current_perm, next_perm);
  }

  // Copy data back to array
  if (its % 2 != 0)
    std::ranges::copy(buffer, range.begin());
}

/// @brief Compute the permutation array that sorts a 2D array by row.
///
/// @param[in] x The flattened 2D array to compute the permutation array
/// for (row-major storage).
/// @param[in] shape1 The number of columns of `x`.
/// @return The permutation array such that `x[perm[i]] <= x[perm[i
/// +1]]`.
/// @pre `x.size()` must be a multiple of `shape1`.
/// @note This function is suitable for small values of `shape1`. Each
/// column of `x` is copied into an array that is then sorted.
template <typename T, int BITS = 16>
std::vector<std::int32_t> sort_by_perm(std::span<const T> x, std::size_t shape1)
{
  static_assert(std::is_integral_v<T>, "Integral required.");

  if (x.empty())
    return std::vector<std::int32_t>{};

  assert(shape1 > 0);
  assert(x.size() % shape1 == 0);
  const std::size_t shape0 = x.size() / shape1;
  std::vector<std::int32_t> perm(shape0);
  std::iota(perm.begin(), perm.end(), 0);

  // Sort by each column, right to left. Col 0 has the most significant
  // "digit".
  std::vector<T> column(shape0);
  for (std::size_t i = 0; i < shape1; ++i)
  {
    std::size_t col = shape1 - 1 - i;
    for (std::size_t j = 0; j < shape0; ++j)
      column[j] = x[j * shape1 + col];
    radix_sort<BITS>(perm, [column = std::cref(column)](auto index)
                     { return column.get()[index]; });
  }

  return perm;
}

} // namespace dolfinx
