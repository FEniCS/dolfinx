// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace dolfinx
{
template <typename T>
void radix_sort(std::vector<T>& array)
{
  if (array.size() <= 1)
    return;

  T max_value = *std::max_element(array.begin(), array.end());

  // Sort 4 bits at a time
  T mask = 0xF;
  constexpr std::size_t bucket_size = 16;

  int passes = 0;
  while (max_value)
  {
    max_value >>= 4;
    passes++;
  }

  std::int32_t mask_offset = 0;
  std::vector<T> buffer(array.size());
  for (int p = 0; p < passes; p++)
  {
    // Count the number of elements in each bucket
    std::int32_t count[bucket_size] = {0};
    std::int32_t offset[bucket_size + 1];
    if (p % 2 == 0)
      for (std::size_t i = 0; i < array.size(); i++)
        count[(array[i] & mask) >> mask_offset]++;
    else
      for (std::size_t i = 0; i < buffer.size(); i++)
        count[(buffer[i] & mask) >> mask_offset]++;

    offset[0] = 0;
    for (std::size_t i = 0; i < bucket_size; i++)
      offset[i + 1] = offset[i] + count[i];

    // now for each element in [lo, hi), move it to its offset in the other
    // buffer this branch should be ok because whichBuf is the same on all
    // threads
    if (p % 2 == 0)
      for (std::size_t i = 0; i < array.size(); i++)
      {
        std::int32_t bucket = (array[i] & mask) >> mask_offset;
        std::int32_t pos = offset[bucket + 1] - count[bucket];
        buffer[pos] = array[i];
        count[bucket]--;
      }
    else
      for (std::size_t i = 0; i < buffer.size(); i++)
      {
        std::int32_t bucket = (buffer[i] & mask) >> mask_offset;
        std::int32_t pos = offset[bucket + 1] - count[bucket];
        array[pos] = buffer[i];
        count[bucket]--;
      }
    mask = mask << 4;
    mask_offset += 4;
  }

  // Move data back to array
  if (passes % 2 != 0)
    std::copy(buffer.begin(), buffer.end(), array.begin());
}
} // namespace dolfinx