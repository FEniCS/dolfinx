// Copyright (C) 2019-2026 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>

/// @file types.h
/// @brief Small, foundational mesh types (enums, etc.) with minimal
/// dependencies.

namespace dolfinx::mesh
{
/// Enum for different partitioning ghost modes
enum class GhostMode : std::uint8_t
{
  none,
  shared_facet
};

} // namespace dolfinx::mesh
