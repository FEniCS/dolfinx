// Copyright (C) 2007-2019 Garth N. Wells
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

/// @brief Iterate over cells and insert entries into sparsity pattern
///
/// @param[in,out] pattern The sparsity pattern to insert into
/// @param[in] topology The mesh topology to build the sparsity pattern
/// over
/// @param[in] dofmaps The dofmap to use in building the sparsity
/// pattern
/// @note The sparsity pattern is not finalised
void cells(
    la::SparsityPattern& pattern, const mesh::Topology& topology,
    const std::array<const std::reference_wrapper<const DofMap>, 2>& dofmaps);

/// @brief Iterate over cells and insert entries into sparsity pattern
///
/// @param[in,out] pattern The sparsity pattern to insert into
/// @param[in] cells The cell indices
/// @param[in] dofmaps The dofmap to use in building the sparsity
/// pattern
/// @note The sparsity pattern is not finalised
void cells(
    la::SparsityPattern& pattern, const std::span<const std::int32_t>& cells,
    const std::array<const std::reference_wrapper<const DofMap>, 2>& dofmaps);

/// @brief Iterate over interior facets and insert entries into sparsity
/// pattern.
///
/// @param[in,out] pattern The sparsity pattern to insert into
/// @param[in] topology The mesh topology to build the sparsity pattern
/// over
/// @param[in] dofmaps The dofmap to use in building the sparsity
/// pattern
/// @note The sparsity pattern is not finalised
void interior_facets(
    la::SparsityPattern& pattern, const mesh::Topology& topology,
    const std::array<const std::reference_wrapper<const DofMap>, 2>& dofmaps);

/// @brief Iterate over interior facets and insert entries into sparsity
/// pattern.
///
/// @param[in,out] pattern The sparsity pattern to insert into
/// @param[in] facets The facets as `(cell, local_index)` pairs, where
/// `local_index` is the index of the facet relative to the `cell`
/// @param[in] dofmaps The dofmap to use in building the sparsity
/// pattern
/// @note The sparsity pattern is not finalised
void interior_facets(
    la::SparsityPattern& pattern, const std::span<const std::int32_t>& facets,
    const std::array<const std::reference_wrapper<const DofMap>, 2>& dofmaps);

/// @brief Iterate over exterior facets and insert entries into sparsity
/// pattern.
///
/// @param[in,out] pattern The sparsity pattern to insert into
/// @param[in] topology The mesh topology to build the sparsity pattern
/// over
/// @param[in] dofmaps The dofmap to use in building the sparsity
/// pattern
/// @note The sparsity pattern is not finalised
void exterior_facets(
    la::SparsityPattern& pattern, const mesh::Topology& topology,
    const std::array<const std::reference_wrapper<const DofMap>, 2>& dofmaps);

/// @brief Iterate over exterior facets and insert entries into sparsity
/// pattern.
///
/// @param[in,out] pattern The sparsity pattern to insert into
/// @param[in] facets The facets as (cell, local_index) pairs
/// @param[in] dofmaps The dofmap to use in building the sparsity
/// pattern
/// @note The sparsity pattern is not finalised
void exterior_facets(
    la::SparsityPattern& pattern, const std::span<const std::int32_t>& facets,
    const std::array<const std::reference_wrapper<const DofMap>, 2>& dofmaps);

} // namespace sparsitybuild
} // namespace dolfinx::fem
