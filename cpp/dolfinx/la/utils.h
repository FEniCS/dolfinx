// Copyright (C) 2018-2022 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <cstdint>
#include <span>

namespace dolfinx::la
{
/// Norm types
enum class Norm
{
  l1,
  l2,
  linf,
  frobenius
};

/// @brief Matrix accumulate/set concept for functions that can be used
/// in assemblers to accumulate or set values in a matrix.
template <class U, class T>
concept MatSet
    = std::invocable<U, std::span<const std::int32_t>,
                     std::span<const std::int32_t>, std::span<const T>>;
} // namespace dolfinx::la
