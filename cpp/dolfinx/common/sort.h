// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

namespace dolfinx
{

// Sort a vector with radix sorting algorithm.
// The bucket size is determined by the number of bits to sort at a time.
template <typename T, int BITS = 8>
void radix_sort(std::vector<T>& array)
{
  static_assert(std::is_integral<T>(), "This function only sorts integers.");

  if (array.size() <= 1)
    return;

  T max_value = *std::max_element(array.begin(), array.end());

  // Sort N bits at a time
  constexpr int bucket_size = 1 << BITS;
  T mask = (T(1) << BITS) - 1;

  // Compute number of iterations, most significant digit (N bits) of maxvalue
  int its = 0;
  while (max_value)
  {
    max_value >>= BITS;
    its++;
  }

  std::int32_t mask_offset = 0;
  std::vector<T> buffer(array.size());

  std::reference_wrapper<std::vector<T>> current_ref = array;
  std::reference_wrapper<std::vector<T>> next_ref = buffer;

  for (int i = 0; i < its; i++)
  {
    std::vector<T>& current = current_ref.get();
    std::vector<T>& next = next_ref.get();

    // Ajdjacency list for computing insertion position
    std::int32_t counter[bucket_size] = {0};
    std::int32_t offset[bucket_size + 1];

    // Count number of elements per bucket.
    for (std::size_t j = 0; j < current.size(); j++)
      counter[(current[j] & mask) >> mask_offset]++;

    // Prefix sum to get the inserting position
    offset[0] = 0;
    std::partial_sum(counter, counter + bucket_size, offset + 1);

    for (std::size_t j = 0; j < current.size(); j++)
    {
      std::int32_t bucket = (current[j] & mask) >> mask_offset;
      std::int32_t new_pos = offset[bucket + 1] - counter[bucket];
      next[new_pos] = current[j];
      counter[bucket]--;
    }

    mask = mask << BITS;
    mask_offset += BITS;

    std::swap(current_ref, next_ref);
  }

  // Move data back to array
  if (its % 2 != 0)
    std::copy(buffer.begin(), buffer.end(), array.begin());
}

// Returns the indices that would sort (lexicographic) a vector of bitsets.
template <int N, int BITS = 8>
std::vector<std::int32_t>
argsort_radix(const std::vector<std::bitset<N>>& array)
{

  std::vector<std::int32_t> perm1(array.size());
  std::iota(perm1.begin(), perm1.end(), 0);
  if (array.size() <= 1)
    return perm1;

  std::vector<std::int32_t> perm2(perm1);

  constexpr int bucket_size = 1 << BITS;
  constexpr int its = N / BITS;
  std::bitset<N> mask = (1 << BITS) - 1;
  std::int32_t mask_offset = 0;

  std::reference_wrapper<std::vector<std::int32_t>> current_perm = perm1;
  std::reference_wrapper<std::vector<std::int32_t>> next_perm = perm2;

  for (int i = 0; i < its; i++)
  {
    // Ajdjacency list for computing insertion position
    std::int32_t counter[bucket_size] = {0};
    std::int32_t offset[bucket_size + 1];

    // Count number of elements per bucket.
    for (std::size_t j = 0; j < array.size(); j++)
    {
      auto set = (array[current_perm.get()[j]] & mask) >> mask_offset;
      std::int32_t bucket = set.to_ulong();
      counter[bucket]++;
    }

    // Prefix sum to get the inserting position
    offset[0] = 0;
    std::partial_sum(counter, counter + bucket_size, offset + 1);

    // Sort py perutation
    for (std::size_t j = 0; j < array.size(); j++)
    {
      auto set = (array[current_perm.get()[j]] & mask) >> mask_offset;
      std::int32_t bucket = set.to_ulong();
      std::int32_t pos = offset[bucket + 1] - counter[bucket];
      next_perm.get()[pos] = current_perm.get()[j];
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