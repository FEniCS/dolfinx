// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace dolfinx::common
{

namespace impl
{

template <typename T>
std::size_t memory(const T& /* obj */)
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

constexpr std::integral_constant<std::int64_t, 1> byte;
constexpr std::integral_constant<std::int64_t, 1'024> kilobyte;
constexpr std::integral_constant<std::int64_t, 1'048'576> megabyte;
constexpr std::integral_constant<std::int64_t, 1'073'741'824> gigabyte;
constexpr std::integral_constant<std::int64_t, 1'099'511'627'776> terabyte;

template <typename T, std::int64_t U = 1>
std::conditional_t<U == 1, std::size_t, double>
memory(const T& obj, std::integral_constant<std::int64_t, U> bytes_per_unit)
{
  std::size_t bytes = impl::memory(obj);
  if constexpr (bytes_per_unit == byte)
    return bytes;
  else
    return static_cast<double>(bytes) / bytes_per_unit.value;
}

} // namespace dolfinx::common
