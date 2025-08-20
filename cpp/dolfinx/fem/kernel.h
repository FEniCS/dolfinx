// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/defines.h>
#include <dolfinx/common/types.h>
#include <ufcx.h>

namespace dolfinx::fem::impl
{
/// @brief Kernel C-pointer type.
/// @tparam T scalar type.
/// @tparam U geometry type.
template <dolfinx::scalar T, std::floating_point U = scalar_value_t<T>>
using kernelptr_t = void (*)(T*, const T*, const T*, const U*, const int*,
                             const std::uint8_t*, void*);

/// @brief Kernel callback type.
/// @tparam T scalar type.
/// @tparam U geometry type.
template <dolfinx::scalar T, std::floating_point U = scalar_value_t<T>>
using kern_t = std::function<void(T*, const T*, const T*, const U*, const int*,
                                  const std::uint8_t*, void*)>;

/// @brief Extract correct kernel by type from UFCx integral.
/// @tparam T scalar type of kernel to extract.
/// @tparam U geometry type of kernel to extract.
/// @param integral UFCx integral to retrieve the kernel from.
/// @return Kernel callback.
template <dolfinx::scalar T, std::floating_point U = scalar_value_t<T>>
constexpr kern_t<T, U> extract_kernel(const ufcx_integral* integral)
{
  if constexpr (std::is_same_v<T, float>)
    return integral->tabulate_tensor_float32;
  else if constexpr (std::is_same_v<T, double>)
    return integral->tabulate_tensor_float64;
  else if constexpr (std::is_same_v<T, std::complex<float>>
                     && has_complex_ufcx_kernels())
  {
    return reinterpret_cast<kernelptr_t<T, U>>(
        integral->tabulate_tensor_complex64);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>
                     && has_complex_ufcx_kernels())
  {
    return reinterpret_cast<kernelptr_t<T, U>>(
        integral->tabulate_tensor_complex128);
  }
  else
    throw std::runtime_error("Could not extract kernel from ufcx integral.");
}

} // namespace dolfinx::fem::impl