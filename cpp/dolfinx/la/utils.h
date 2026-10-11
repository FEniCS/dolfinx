// Copyright (C) 2018-2022 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <format>
#include <ranges>
#include <span>
#include <stdexcept>

namespace dolfinx::la
{
/// Norm types
enum class Norm : std::int8_t
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
/// @brief Set a value on the diagonal of a matrix for specified rows.
///
/// This function is typically called after assembly. The assembly
/// function zeroes Dirichlet rows and columns. For block matrices, this
/// function should normally be called only on the diagonal blocks, i.e.
/// blocks for which the test and trial spaces are the same.
///
/// @note `set_fn` decides whether the value replaces the existing
/// entry or is added to it. A row repeated in `rows` is passed to
/// `set_fn` once per occurrence, which an adding `set_fn` accumulates.
///
/// @param[in] set_fn The function for setting values to a matrix.
/// @param[in] rows Row blocks, in local indices, whose diagonal to
/// set. May have static or dynamic extent.
/// @param[in] diagonal Value for the diagonal of each row in `rows`.
template <dolfinx::scalar T>
void set_diagonal(MatSet<T> auto&& set_fn,
                  const common::LocalIndexRange auto& rows, T diagonal = T(1))
{
  std::span<const T, 1> diag_span(&diagonal, 1);
  for (std::size_t i = 0; i < std::ranges::size(rows); ++i)
  {
    std::span<const std::int32_t, 1> row(std::ranges::data(rows) + i, 1);
    set_fn(row, row, diag_span);
  }
}

/// @brief Set values on the diagonal of a matrix for specified rows,
/// with a value per row.
///
/// See the single-value set_diagonal for usage.
///
/// @param[in] set_fn The function for setting values to a matrix.
/// @param[in] rows Row blocks, in local indices, whose diagonal to
/// set. May have static or dynamic extent.
/// @param[in] diagonals Diagonal values, with `diagonals[i]` the value
/// for `rows[i]`. Must have the same length as `rows`.
template <dolfinx::scalar T>
void set_diagonal(MatSet<T> auto&& set_fn,
                  const common::LocalIndexRange auto& rows,
                  std::span<const T> diagonals)
{
  if (diagonals.size() != std::ranges::size(rows))
  {
    throw std::invalid_argument(
        std::format("Number of diagonal values ({}) does not match number "
                    "of rows ({}).",
                    diagonals.size(), std::ranges::size(rows)));
  }

  for (std::size_t i = 0; i < diagonals.size(); ++i)
  {
    set_diagonal(
        set_fn,
        std::span<const std::int32_t, 1>(std::ranges::data(rows) + i, 1),
        diagonals[i]);
  }
}
} // namespace dolfinx::la
