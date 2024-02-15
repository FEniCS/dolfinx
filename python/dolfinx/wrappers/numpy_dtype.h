
// Copyright (C) 2024 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <complex>
#include <cstdint>
#include <type_traits>

#pragma once

namespace dolfinx_wrappers
{

/// @brief Return NumPy dtype char type for a C++ type
/// @tparam T C++ type
/// @return NumPy dtype
template <typename T>
constexpr char numpy_dtype()
{
  if constexpr (std::is_same_v<T, float>)
    return 'f';
  else if constexpr (std::is_same_v<T, double>)
    return 'd';
  else if constexpr (std::is_same_v<T, std::complex<float>>)
    return 'F';
  else if constexpr (std::is_same_v<T, std::complex<double>>)
    return 'D';
  else if constexpr (std::is_same_v<T, std::int8_t>)
    return 'b';
  else if constexpr (std::is_same_v<T, std::uint8_t>)
    return 'B';
  else if constexpr (std::is_same_v<T, std::int32_t>)
    return 'i';
  else if constexpr (std::is_same_v<T, std::int32_t>)
    return 'I';
  else if constexpr (std::is_same_v<T, std::int64_t>)
    return 'l';
  else if constexpr (std::is_same_v<T, std::int64_t>)
    return 'L';
}
} // namespace dolfinx_wrappers
