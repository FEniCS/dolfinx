// Copyright (C) 2018-2026 Garth N. Wells, Chris N. Richardson and Jørgen S.
// Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <dolfinx/mesh/CoordinateElement.h>

namespace dolfinx::fem
{
/// @deprecated Use mesh::CoordinateElement instead.
template <std::floating_point T>
using CoordinateElement [[deprecated("Use mesh::CoordinateElement instead.")]]
= mesh::CoordinateElement<T>;
} // namespace dolfinx::fem
