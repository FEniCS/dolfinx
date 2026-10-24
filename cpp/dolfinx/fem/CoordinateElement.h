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
// Deprecated: use mesh::CoordinateElement. Hidden from Doxygen, which (in
// 1.15) misparses an attribute on an alias declaration.
/// @cond
template <std::floating_point T>
using CoordinateElement [[deprecated("Use mesh::CoordinateElement instead.")]]
= mesh::CoordinateElement<T>;
/// @endcond
} // namespace dolfinx::fem
