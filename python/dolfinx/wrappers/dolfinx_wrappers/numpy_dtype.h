
// Copyright (C) 2024 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <complex>
#include <cstdint>

#pragma once

namespace dolfinx_wrappers
{

// Primary template: unimplemented (will cause compile error if not specialized)
template <typename T>
struct numpy_dtype_traits;

// Built-in specializations
template <>
struct numpy_dtype_traits<float>
{
  static constexpr char value = 'f';
};
template <>
struct numpy_dtype_traits<double>
{
  static constexpr char value = 'd';
};
template <>
struct numpy_dtype_traits<std::complex<float>>
{
  static constexpr char value = 'F';
};
template <>
struct numpy_dtype_traits<std::complex<double>>
{
  static constexpr char value = 'D';
};
template <>
struct numpy_dtype_traits<std::int8_t>
{
  static constexpr char value = 'b';
};
template <>
struct numpy_dtype_traits<std::uint8_t>
{
  static constexpr char value = 'B';
};
template <>
struct numpy_dtype_traits<std::int32_t>
{
  static constexpr char value = 'i';
};
template <>
struct numpy_dtype_traits<std::uint32_t>
{
  static constexpr char value = 'I';
};
template <>
struct numpy_dtype_traits<std::int64_t>
{
  static constexpr char value = 'l';
};
template <>
struct numpy_dtype_traits<std::uint64_t>
{
  static constexpr char value = 'L';
};

template <typename T>
constexpr char numpy_dtype()
{
  return numpy_dtype_traits<T>::value;
}

} // namespace dolfinx_wrappers
