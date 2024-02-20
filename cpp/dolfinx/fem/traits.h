// Copyright (C) 2024 Joseph P. Dean and Garth N. Wells
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <type_traits>

namespace dolfinx::fem
{

/// @brief Finite element cell kernel concept.
///
/// Kernel functions that can be passed to an assembler for execution
/// must satisfy this concept.
template <class U, class T>
concept FEkernel = std::is_invocable_v<U, T*, const T*, const T*,
                                       const scalar_value_type_t<T>*,
                                       const int*, const std::uint8_t*>;

} // namespace dolfinx::fem
