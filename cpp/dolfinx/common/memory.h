// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dolfinx::common
{

namespace impl
{

template <typename T>
std::size_t memory(const T& obj)
{
  static_assert(false, "Memory usage not supported for provided type.");
}

template <typename T>
std::size_t memory(const std::vector<T>& vec)
{
  using value_type = typename std::vector<T>::value_type;

  std::size_t size_type = sizeof(vec);
  std::size_t size_data = vec.capacity() * sizeof(value_type);
  return size_type + size_data;
}

} // namespace impl

constexpr std::int64_t byte = 1;
constexpr std::int64_t kilobyte = 1024;
constexpr std::int64_t megabyte = 1'048'576;
constexpr std::int64_t gigabyte = 1'073'741'824;
constexpr std::int64_t terabyte = 1'099'511'627'776;

template <typename T>
double memory(const T& obj, std::int64_t bytes_per_unit = byte)
{
  std::size_t bytes = impl::memory(obj);
  return static_cast<double>(bytes) / bytes_per_unit;
}

} // namespace dolfinx::common
