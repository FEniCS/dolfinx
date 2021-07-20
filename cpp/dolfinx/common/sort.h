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
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

namespace
{
template <typename T>
std::int32_t to_int(T a)
{
  if constexpr (std::is_integral<T>())
    return a;
  else
    return a.to_ulong();
}
} // namespace
namespace dolfinx
{

/// Sort a vector of integers with radix sorting algorithm.The bucket size is
/// determined by the number of bits to sort at a time (2^BITS).
/// @tparam T Integral type
/// @tparam BITS The number of bits to sort at a time.
/// @param[in, out] array The array to sort.
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

/// Returns the indices that would sort (lexicographic) a vector of bitsets.
/// @tparam N The size of the bitset, which corresponds to the number of bits
/// necessary to represent a set of integers. For example, N = 96 for mapping
/// three std::int32_t.
/// @tparam BITS The number of bits to sort at a time.
/// @param[in] array The array to sort.
/// Returns Vector of indices that sort the input array.
template <typename T, int BITS = 8>
std::vector<std::int32_t> argsort_radix(const xtl::span<const T>& array)
{
  std::vector<std::int32_t> perm1(array.size());
  std::iota(perm1.begin(), perm1.end(), 0);
  if (array.size() <= 1)
    return perm1;

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

  if (its % 2 == 0)
    return perm1;
  else
    return perm2;
}

/// Returns the indices that would sort (lexicographic) a vector of bitsets.
/// @tparam N The size of the bitset, which corresponds to the number of bits
/// necessary to represent a set of integers. For example, N = 96 for mapping
/// three std::int32_t.
/// @tparam BITS The number of bits to sort at a time.
/// @param[in] array The array to sort.
/// Returns Vector of indices that sort the input array.
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
namespace
{
template <std::size_t d>
std::vector<std::int32_t>
sort_by_perm_impl(const xt::xtensor<std::int32_t, 2>& array)
{
  assert(array.shape(1) == d);
  constexpr int set_size = 32 * d;
  std::vector<std::bitset<set_size>> bit_array(array.shape(0));

  // Pack list of "d" ints into a bitset
  for (std::size_t i = 0; i < array.shape(0); i++)
  {
    for (std::size_t j = 0; j < d; j++)
    {
      std::bitset<set_size> bits = array(i, j);
      bit_array[i] |= bits << (32 * (d - j - 1));
    }
  }

  // This function creates a 2**16 temporary array (buckets)
  return dolfinx::argsort_radix<set_size, 16>(bit_array);
}
} // namespace

namespace dolfinx
{
/// Takes an array and computes the sort permutation that would reorder
/// the rows in ascending order
/// @tparam The length of each row of @p array
/// @param[in] array The input array
/// @return The permutation vector that would order the rows in
/// ascending order
/// @pre Each row of @p array must be sorted
template <typename T>
std::vector<std::int32_t> sort_by_perm(const xt::xtensor<T, 2>& array)
{
  // Sort the list and label uniquely
  std::vector<std::int32_t> sort_order;
  const int num_vert = array.shape(1);

  switch (num_vert)
  {
  case 2:
    return sort_by_perm_impl<2>(array);
  case 3:
    return sort_by_perm_impl<3>(array);
  case 4:
    return sort_by_perm_impl<4>(array);
  case 5:
    return sort_by_perm_impl<5>(array);
  default:
    throw std::runtime_error("Not implemented for this shape.");
    break;
  }
  return sort_order;
}

template <typename T, int BITS = 16>
std::vector<std::int32_t> sort_by_perm_new(const xt::xtensor<T, 2>& array)
{
  // Sort the list and label uniquely
  const int cols = array.shape(1);
  const int size = array.shape(0);
  std::vector<std::int32_t> perm(size);
  std::vector<std::int32_t> local_perm(size);
  std::iota(perm.begin(), perm.end(), 0);

  for (int i = 0; i < cols; i++)
  {
    int col = cols - 1 - i;
    xt::xtensor<std::int32_t, 1> column = xt::view(array, xt::keep(perm), col);
    local_perm = argsort_radix(xtl::span<const std::int32_t>(column));

    // Swap to avoid temp
    std::swap(perm, local_perm);
    for (int j = 0; j < size; j++)
      perm[j] = local_perm[perm[j]];
  }

  return perm;
}

} // namespace dolfinx
