// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <type_traits>

namespace dolfinx::refinement
{
/// @brief Options for data to compute during mesh refinement.
enum class Option : std::uint8_t
{
  none = 0b00,         /*!< No extra data */
  parent_facet = 0b01, /*!< Compute list of the cell-local facet indices in the
          parent cell of each facet in each new cell (or -1 if no match)  */
  parent_cell
  = 0b10, /*!< Compute list with the parent cell index for each new cell */
  parent_cell_and_facet = 0b11 /*< Both cell and facet parent data */
};

/// @brief Combine two refinement options into one, both flags will be
/// set for the resulting option.
inline constexpr Option operator|(Option a, Option b)
{
  using bitmask_t = std::underlying_type_t<Option>;
  bitmask_t a_native = static_cast<bitmask_t>(a);
  bitmask_t b_native = static_cast<bitmask_t>(b);
  return static_cast<Option>(a_native | b_native);
}

/// @brief Check if parent_facet flag is set
inline constexpr bool option_parent_facet(Option a)
{
  using bitmask_t = std::underlying_type_t<Option>;
  bitmask_t a_native = static_cast<bitmask_t>(a);
  bitmask_t facet_native = static_cast<bitmask_t>(Option::parent_facet);
  return (a_native & facet_native) == facet_native;
}

/// @brief Check if parent_cell flag is set
inline constexpr bool option_parent_cell(Option a)
{
  using bitmask_t = std::underlying_type_t<Option>;
  bitmask_t a_native = static_cast<bitmask_t>(a);
  bitmask_t facet_native = static_cast<bitmask_t>(Option::parent_cell);
  return (a_native & facet_native) == facet_native;
}
} // namespace dolfinx::refinement
