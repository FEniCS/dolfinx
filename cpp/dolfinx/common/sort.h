// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <dolfinx/common/Timer.h>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

namespace dolfinx
{

/// Sort a vector of integers with radix sorting algorithm. The bucket
/// size is determined by the number of bits to sort at a time (2^BITS).
/// @tparam T Integral type
/// @tparam BITS The number of bits to sort at a time.
/// @param[in, out] array The array to sort.
template <typename T, int BITS = 8>
void radix_sort(std::span<T> array)
{
  static_assert(std::is_integral<T>(), "This function only sorts integers.");

  if (array.size() <= 1)
    return;

  T max_value = *std::max_element(array.begin(), array.end());

  // Sort N bits at a time
  constexpr int bucket_size = 1 << BITS;
  T mask = (T(1) << BITS) - 1;

  // Compute number of iterations, most significant digit (N bits) of
  // maxvalue
  int its = 0;
  while (max_value)
  {
    max_value >>= BITS;
    its++;
  }

  // Adjacency list arrays for computing insertion position
  std::array<std::int32_t, bucket_size> counter;
  std::array<std::int32_t, bucket_size + 1> offset;

  std::int32_t mask_offset = 0;
  std::vector<T> buffer(array.size());
  std::span<T> current_perm = array;
  std::span<T> next_perm = buffer;
  for (int i = 0; i < its; i++)
  {
    // Zero counter array
    std::fill(counter.begin(), counter.end(), 0);

    // Count number of elements per bucket
    for (T c : current_perm)
      counter[(c & mask) >> mask_offset]++;

    // Prefix sum to get the inserting position
    offset[0] = 0;
    std::partial_sum(counter.begin(), counter.end(), std::next(offset.begin()));
    for (T c : current_perm)
    {
      std::int32_t bucket = (c & mask) >> mask_offset;
      std::int32_t new_pos = offset[bucket + 1] - counter[bucket];
      next_perm[new_pos] = c;
      counter[bucket]--;
    }

    mask = mask << BITS;
    mask_offset += BITS;

    std::swap(current_perm, next_perm);
  }

  // Copy data back to array
  if (its % 2 != 0)
    std::copy(buffer.begin(), buffer.end(), array.begin());
}

/// Returns the indices that would sort (lexicographic) a vector of
/// bitsets.
/// @tparam T The size of the bitset, which corresponds to the number of
/// bits necessary to represent a set of integers. For example, N = 96
/// for mapping three std::int32_t.
/// @tparam BITS The number of bits to sort at a time
/// @param[in] array The array to sort
/// @param[in] perm FIXME
template <typename T, int BITS = 16>
void argsort_radix(std::span<const T> array, std::span<std::int32_t> perm)
{
  static_assert(std::is_integral_v<T>, "Integral required.");

  if (array.size() <= 1)
    return;

  const auto [min, max] = std::minmax_element(array.begin(), array.end());
  T range = *max - *min + 1;

  // Sort N bits at a time
  constexpr int bucket_size = 1 << BITS;
  T mask = (T(1) << BITS) - 1;
  std::int32_t mask_offset = 0;

  // Compute number of iterations, most significant digit (N bits) of
  // maxvalue
  int its = 0;
  while (range)
  {
    range >>= BITS;
    its++;
  }

  // Adjacency list arrays for computing insertion position
  std::array<std::int32_t, bucket_size> counter;
  std::array<std::int32_t, bucket_size + 1> offset;

  std::vector<std::int32_t> perm2(perm.size());
  std::span<std::int32_t> current_perm = perm;
  std::span<std::int32_t> next_perm = perm2;
  for (int i = 0; i < its; i++)
  {
    // Zero counter
    std::fill(counter.begin(), counter.end(), 0);

    // Count number of elements per bucket
    for (auto cp : current_perm)
    {
      T value = array[cp] - *min;
      std::int32_t bucket = (value & mask) >> mask_offset;
      counter[bucket]++;
    }

    // Prefix sum to get the inserting position
    offset[0] = 0;
    std::partial_sum(counter.begin(), counter.end(), std::next(offset.begin()));

    // Sort py permutation
    for (auto cp : current_perm)
    {
      T value = array[cp] - *min;
      std::int32_t bucket = (value & mask) >> mask_offset;
      std::int32_t pos = offset[bucket + 1] - counter[bucket];
      next_perm[pos] = cp;
      counter[bucket]--;
    }

    std::swap(current_perm, next_perm);

    mask = mask << BITS;
    mask_offset += BITS;
  }

  if (its % 2 == 1)
    std::copy(perm2.begin(), perm2.end(), perm.begin());
}

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
    argsort_radix<T, BITS>(column, perm);
  }

  return perm;
}

} // namespace dolfinx
