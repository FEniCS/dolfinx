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
#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx
{

/// Sort a vector of integers with radix sorting algorithm.The bucket size is
/// determined by the number of bits to sort at a time (2^BITS).
/// @tparam T Integral type
/// @tparam BITS The number of bits to sort at a time.
/// @param[in, out] array The array to sort.
template <typename T, int BITS = 8>
void radix_sort(const xtl::span<T>& array)
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
  xtl::span<T> current_perm = array;
  xtl::span<T> next_perm = buffer;
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
void argsort_radix(const xtl::span<const T>& array,
                   xtl::span<std::int32_t> perm)
{
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
  xtl::span<std::int32_t> current_perm = perm;
  xtl::span<std::int32_t> next_perm = perm2;
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

template <typename T, int BITS = 16>
std::vector<std::int32_t> sort_by_perm(const xt::xtensor<T, 2>& array)
{
  // Sort the list and label uniquely
  const int cols = array.shape(1);
  const int size = array.shape(0);
  std::vector<std::int32_t> perm(size);
  std::iota(perm.begin(), perm.end(), 0);

  // Sort each column at a time from right to left.
  // Col 0 has the most signficant "digit".
  for (int i = 0; i < cols; i++)
  {
    int col = cols - 1 - i;
    xt::xtensor<std::int32_t, 1> column = xt::view(array, xt::all(), col);
    argsort_radix<std::int32_t, BITS>(xtl::span<const std::int32_t>(column),
                                      perm);
  }

  return perm;
}

} // namespace dolfinx
