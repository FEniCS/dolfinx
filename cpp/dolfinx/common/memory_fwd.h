// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>

namespace dolfinx::mesh
{
template <std::floating_point T>
class Geometry;
}

namespace dolfinx::common::impl
{
template <std::floating_point T>
std::size_t memory(const dolfinx::mesh::Geometry<T>& geometry);
}