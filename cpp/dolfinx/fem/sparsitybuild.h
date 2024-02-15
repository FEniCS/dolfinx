// Copyright (C) 2007-2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <span>

namespace dolfinx::la
{
class SparsityPattern;
}

namespace dolfinx::mesh
{
class Topology;
}

namespace dolfinx::fem
{
class DofMap;

/// Support for building sparsity patterns from degree-of-freedom maps.
namespace sparsitybuild
{
/// @brief Iterate over cells and insert entries into sparsity pattern.
///
/// @param[in,out] pattern The sparsity pattern to insert into
/// @param[in] cells The cell indices
/// @param[in] dofmaps Dofmaps to used in building the sparsity pattern
/// @note The sparsity pattern is not finalised
void cells(la::SparsityPattern& pattern, std::span<const std::int32_t> cells,
           std::array<std::reference_wrapper<const DofMap>, 2> dofmaps);

/// @brief TODO
/// @param pattern TODO
/// @param cells_0 TODO
/// @param cells_1 TODO
/// @param dofmaps TODO
void cells(la::SparsityPattern& pattern, std::span<const std::int32_t> cells_0,
           std::span<const std::int32_t> cells_1,
           std::array<std::reference_wrapper<const DofMap>, 2> dofmaps);

/// @brief Iterate over interior facets and insert entries into sparsity
/// pattern.
///
/// @param[in,out] pattern Sparsity pattern to insert into
/// @param[in] facets Facets as `(cell0, cell1)` pairs for each facet.
/// @param[in] dofmaps The dofmap to use in building the sparsity
/// pattern.
/// @note The sparsity pattern is not finalised.
void interior_facets(
    la::SparsityPattern& pattern, std::span<const std::int32_t> facets,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps);

} // namespace sparsitybuild
} // namespace dolfinx::fem
