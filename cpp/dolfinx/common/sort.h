// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <numeric>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx
{

// Sort a vector with radix sorting algorithm. The bucket size is
// determined by the number of bits to sort at a time.
template <typename T, int BITS = 8>
void radix_sort(xtl::span<T> array)
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

// Returns the indices that would sort (lexicographic) a vector of
// bitsets
template <int N, int BITS = 8>
std::vector<std::int32_t>
argsort_radix(const xtl::span<const std::bitset<N>>& array)
{
  std::vector<std::int32_t> perm1(array.size());
  std::iota(perm1.begin(), perm1.end(), 0);
  if (array.size() <= 1)
    return perm1;

  constexpr int bucket_size = 1 << BITS;
  constexpr int its = N / BITS;
  std::bitset<N> mask = (1 << BITS) - 1;
  std::int32_t mask_offset = 0;

  // Adjacency list arrays for computing insertion position
  std::array<std::int32_t, bucket_size> counter;
  std::array<std::int32_t, bucket_size + 1> offset;

  std::vector<std::int32_t> perm2 = perm1;
  xtl::span<std::int32_t> current_perm = perm1;
  xtl::span<std::int32_t> next_perm = perm2;
  for (int i = 0; i < its; i++)
  {
    // Zero counter
    std::fill(counter.begin(), counter.end(), 0);

    // Count number of elements per bucket
    for (auto cp : current_perm)
    {
      auto set = (array[cp] & mask) >> mask_offset;
      auto bucket = set.to_ulong();
      counter[bucket]++;
    }

    // Prefix sum to get the inserting position
    offset[0] = 0;
    std::partial_sum(counter.begin(), counter.end(), std::next(offset.begin()));

    // Sort py permutation
    for (auto cp : current_perm)
    {
      auto set = (array[cp] & mask) >> mask_offset;
      auto bucket = set.to_ulong();
      std::int32_t pos = offset[bucket + 1] - counter[bucket];
      next_perm[pos] = cp;
      counter[bucket]--;
    }

    std::swap(current_perm, next_perm);

    mask = mask << BITS;
    mask_offset += BITS;
  }

  if (its % 2 == 0)
    return perm1;
  else
    return perm2;
}
} // namespace dolfinx
