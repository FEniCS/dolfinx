// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>

namespace dolfinx
{

namespace common
{
class IndexMap;
}

namespace mesh
{
template <std::floating_point T>
class Geometry;
}

namespace common::impl
{
std::size_t memory(const IndexMap& im);

template <std::floating_point T>
std::size_t memory(const dolfinx::mesh::Geometry<T>& geometry);
} // namespace common::impl

} // namespace dolfinx