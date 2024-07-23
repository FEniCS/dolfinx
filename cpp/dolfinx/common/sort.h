// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace dolfinx
{

struct __radix_sort
{

  /// @brief Sort a range with radix sorting algorithm. The bucket
  /// size is determined by the number of bits to sort at a time (2^BITS).
  ///
  /// This allows usage with standard range containers of integral types, for
  /// example
  /// @code
  /// std::array<std::int16_t, 3> a{2, 3, 1};
  /// dolfixn::radix_sort(a); // a = {1, 2, 3}
  /// @endcode
  /// Additionally the projection based approach of the STL library is adpated,
  /// which allows for versatile usage, for example the easy realization of an
  /// argsort
  /// @code
  /// std::array<std::int16_t, 3> a{2, 3, 1};
  /// std::array<std::int16_t, 3> i{0, 1, 2};
  /// dolfixn::radix_sort(i, [&](auto i){ return a[i]; }); // yields i = {2, 0,
  /// 1} and a[i] = {1, 2, 3};
  /// @endcode
  /// @tparam R Type of range to be sorted.
  /// @tparam P Projection type to be applied on range elements to produce a
  /// sorting index.
  /// @tparam BITS The number of bits to sort at a time.
  /// @param[in, out] range The range to sort.
  /// @param[in] P Element projection.
  template <
      std::ranges::random_access_range R, typename P = std::identity,
      std::remove_cvref_t<std::invoke_result_t<P, std::iter_value_t<R>>> BITS
      = 8>
    requires std::integral<
        std::remove_cvref_t<std::invoke_result_t<P, std::iter_value_t<R>>>>
  constexpr void operator()(R&& range, P proj = {}) const
  {
    // value type
    using T = std::iter_value_t<R>;

    // index type (if no projection is provided it holds I == T)
    using I = std::remove_cvref_t<std::invoke_result_t<P, T>>;

    if (range.size() <= 1)
      return;

    T max_value = proj(*std::ranges::max_element(range, std::less{}, proj));

    // Sort N bits at a time
    constexpr I bucket_size = 1 << BITS;
    T mask = (T(1) << BITS) - 1;

    // Compute number of iterations, most significant digit (N bits) of
    // maxvalue
    I its = 0;
    while (max_value)
    {
      max_value >>= BITS;
      its++;
    }

    // Adjacency list arrays for computing insertion position
    std::array<I, bucket_size> counter;
    std::array<I, bucket_size + 1> offset;

    I mask_offset = 0;
    std::vector<T> buffer(range.size());
    std::span<T> current_perm = range;
    std::span<T> next_perm = buffer;
    for (I i = 0; i < its; i++)
    {
      // Zero counter array
      std::ranges::fill(counter, 0);

      // Count number of elements per bucket
      for (const auto& c : current_perm)
        counter[(proj(c) & mask) >> mask_offset]++;

      // Prefix sum to get the inserting position
      offset[0] = 0;
      std::partial_sum(counter.begin(), counter.end(),
                       std::next(offset.begin()));
      for (const auto& c : current_perm)
      {
        I bucket = (proj(c) & mask) >> mask_offset;
        I new_pos = offset[bucket + 1] - counter[bucket];
        next_perm[new_pos] = c;
        counter[bucket]--;
      }

      mask = mask << BITS;
      mask_offset += BITS;

      std::swap(current_perm, next_perm);
    }

    // Copy data back to array
    if (its % 2 != 0)
      std::ranges::copy(buffer, range.begin());
  }
};

/// Radix sort
inline constexpr __radix_sort radix_sort{};

/// @brief Compute the permutation array that sorts a 2D array by row.
///
/// @param[in] x The flattened 2D array to compute the permutation array
/// for.
/// @param[in] shape1 The number of columns of `x`.
/// @return The permutation array such that `x[perm[i]] <= x[perm[i +1]].
/// @pre `x.size()` must be a multiple of `shape1`.
/// @note This function is suitable for small values of `shape1`. Each
/// column of `x` is copied into an array that is then sorted.
template <typename T, int BITS = 16>
std::vector<std::int32_t> sort_by_perm(std::span<const T> x, std::size_t shape1)
{
  static_assert(std::is_integral_v<T>, "Integral required.");
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
    int col = shape1 - 1 - i;
    for (std::size_t j = 0; j < shape0; ++j)
      column[j] = x[j * shape1 + col];

    radix_sort(perm, [&column](auto index) { return column[index]; });
  }

  return perm;
}

} // namespace dolfinx
