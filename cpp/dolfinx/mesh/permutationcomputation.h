// Copyright (C) 2020 Matthew Scroggs
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace dolfinx::mesh
{
class Topology;

/// Compute (1) facet rotation and reflection data, and (2) cell
/// permutation data. This information is used assemble of (1) facet
/// inetgrals and (2) vector elements.
///
/// 1. Get the permutation numbers to apply to facets. The
///    permutations are numbered so that:
///
///     - `n % 2` gives the number of reflections to apply
///     - `n // 2` gives the number of rotations to apply
///
///    Each column of the returned array represents a cell, and each
///    row a facet of that cell. This data is used to permute the
///    quadrature point on facet integrals when data from the cells on
///    both sides of the facet is used.
///
/// 2. Get the permutation information about the entities of each
///    cell, relative to a low-to-high ordering. This data is packed
///    so that a 32 bit int is used for each cell. For 2D cells, one
///    bit is used for each edge, to represent whether or not the edge
///    is reversed: the least significant bit is for edge 0, the next
///    for edge 1, etc. For 3D cells, three bits are used for each
///    face, and for each edge: the least significant bit says whether
///    or not face 0 is reflected, the next 2 bits say how many times
///    face 0 is rotated; the next three bits are for face 1, then
///    three for face 2, etc; after all the faces, there is 1 bit for
///    each edge to say whether or not they are reversed.
///
///    For example, if a quadrilateral has cell permutation info
///    `....0111` then (from right to left):
///
///      - edge 0 is reflected (1)
///      - edge 1 is reflected (1)
///      - edge 2 is reflected (1)
///      - edge 3 is not permuted (0)
///
///    and if a tetrahedron has cell permutation info
///    `....011010010101001000` then (from right to left):
///
///      - face 0 is not permuted (000)
///      - face 1 is reflected (001)
///      - face 2 is rotated twice then reflected (101)
///      - face 3 is rotated once (010)
///      - edge 0 is not permuted (0)
///      - edge 1 is reflected (1)
///      - edge 2 is not permuted (0)
///      - edge 3 is reflected (1)
///      - edge 4 is reflected (1)
///      - edge 5 is not permuted (0)
///
///    This data is used to correct the direction of vector function
///    on permuted facets.
///
/// @return Facet permutation and cells permutations
std::pair<std::vector<std::uint8_t>, std::vector<std::uint32_t>>
compute_entity_permutations(const Topology& topology);


/// @brief TODO
/// @todo: Rename
std::vector<std::uint8_t> compute_cell_permutations(const Topology& topology);

} // namespace dolfinx::mesh
