// Copyright (C) 2020 Matthew Scroggs
// Copyright (C) 2020-2026 Matthew Scroggs and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace dolfinx::mesh
{
class Topology;

/// @brief Bit of the cell permutation info that negates the
/// degrees-of-freedom on facet `f` of a cell.
///
/// Set by Topology::create_facet_orientations. An element mapped by the
/// contravariant Piola map is conforming only between cells that agree
/// on the normal of the facet they share; where they do not, one of the
/// two negates its degrees-of-freedom there. A facet's
/// degrees-of-freedom are the only ones whose normal trace is non-zero
/// on it, so this changes that trace and no other. See
/// compute_facet_orientations.
///
/// Facets of a cell of topological dimension 2 are its edges, of which
/// there are at most four, so bits 27 to 30 suffice. The sub-entity
/// permutations packed into the same integer (see
/// compute_cell_permutations) reach bit 29 for a hexahedron, but a cell
/// carrying facet signs has topological dimension 2 and so uses one bit
/// per edge. These bits are cleared (`info & ~orientation_mask(tdim)`)
/// before the info is passed to Basix, which knows only the sub-entity
/// permutations.
///
/// @param[in] f Cell-local facet index, `0 <= f < 4`.
constexpr std::uint32_t reversed_facet_bit(int f)
{
  return std::uint32_t(1) << (27 + f);
}

/// @brief Bits of the cell permutation info that hold the facet signs of
/// a cell of topological dimension `tdim`.
///
/// Only a surface carries facet signs. The sub-entity permutations of a
/// hexahedron already reach bit 29 (six faces at three bits and twelve
/// edges at one), so there is no room for them there, and none is
/// needed: an orientation exists only for `tdim == 2`.
///
/// @param[in] tdim Topological dimension of the cell.
constexpr std::uint32_t orientation_mask(int tdim)
{
  if (tdim == 2)
  {
    return reversed_facet_bit(0) | reversed_facet_bit(1) | reversed_facet_bit(2)
           | reversed_facet_bit(3);
  }
  return 0;
}

/// @brief Compute the permutation to apply to each cell-local entity of
/// a given dimension.
///
/// The permutation is encoded so that:
///
///  - `n % 2` gives the number of reflections to apply
///  - `n // 2` gives the number of rotations to apply
///
/// The data is stored in a flattened 2D array, so that
/// `data[cell_index * entities_per_cell + entity_index]` contains the
/// permutation data for the entity with index `entity_index` of cell
/// `cell_index`. This data passed to FFCx kernels, where it is used to
/// permute quadrature points on sub-entity integrals when data from
/// more than one cell incident to the entity is used.

///
/// @param[in] topology Mesh topology.
/// @param[in] dim Topological dimension of the entities to permute.
/// Must satisfy `0 <= dim < topology.dim()`. Vertices (`dim == 0`) have
/// no orientation, so an empty vector is returned for them.
/// @param[in] num_threads Number of threads to use.
/// @return Permutation of each cell-local entity of dimension `dim`,
/// flattened row-wise.
/// @see compute_cell_permutations, which packs the orientations of all
/// of a cell's sub-entities into one integer per cell, for correcting
/// element DOFs rather than quadrature points.
std::vector<std::uint8_t> compute_entity_permutations(const Topology& topology,
                                                      int dim, int num_threads);

/// @brief Compute the packed per-cell permutation data.
///
/// Required by elements whose DOF transformations are not the identity.
/// Where those transformations are permutations, e.g. higher-order
/// Lagrange, the correction is applied once to the dofmap when it is
/// built; otherwise, e.g. N1curl and Raviart-Thomas, the correction
/// is applied to the element tensor on each cell at assembly time.
///
/// The cell permutation data contains information about the entities of
/// each cell, relative to a low-to-high ordering. This data is packed
/// so that a 32-bit int is used for each cell. For 2D cells, one bit is
/// used for each edge, to represent whether or not the edge is
/// reversed: the least significant bit is for edge 0, the next for edge
/// 1, etc. For 3D cells, three bits are used for each face, and for
/// each edge: the least significant bit says whether or not face 0 is
/// reflected, the next 2 bits say how many times face 0 is rotated; the
/// next three bits are for face 1, then three for face 2, etc; after
/// all the faces, there is 1 bit for each edge to say whether or not
/// they are reversed. On a surface, bits 27 to 30 are left for the
/// facet signs (see ::reversed_facet_bit).
///
/// For example, if a quadrilateral has cell permutation info
/// `....0111` then (from right to left):
///
///   - edge 0 is reflected (1)
///   - edge 1 is reflected (1)
///   - edge 2 is reflected (1)
///   - edge 3 is not permuted (0)
///
/// and if a tetrahedron has cell permutation info
/// `....011010010101001000` then (from right to left):
///
///   - face 0 is not permuted (000)
///   - face 1 is reflected (001)
///   - face 2 is rotated twice then reflected (101)
///   - face 3 is rotated once (010)
///   - edge 0 is not permuted (0)
///   - edge 1 is reflected (1)
///   - edge 2 is not permuted (0)
///   - edge 3 is reflected (1)
///   - edge 4 is reflected (1)
///   - edge 5 is not permuted (0)
///
/// @note Not collective.
/// @pre All entities of dimension `< topology.dim()` must already exist
/// (see Topology::create_entities).
/// @param[in] topology Mesh topology.
/// @param[in] num_threads Number of threads to use. Must be >= 1.
/// @return Facet permutation and cell permutations, covering both owned
/// and ghost cells.
/// @throws std::invalid_argument If `num_threads < 1`.
/// @throws std::runtime_error If `topology` is a mixed-topology mesh
/// (more than one 3D cell type).
std::vector<std::uint32_t> compute_cell_permutations(const Topology& topology,
                                                     int num_threads);

/// @brief Find the facets whose two cells disagree on the normal.
///
/// An element mapped by the contravariant Piola map is conforming only
/// between cells that agree on the normal of the facet they share. The
/// vertex order of a cell fixes its normal, and two cells agree exactly
/// when they run their shared facet in opposite directions.
///
/// Where they do not, one of the two carries a sign on that facet.
/// Negating its degrees-of-freedom there restores the agreement and
/// disturbs nothing else: of a cell's basis functions only those
/// belonging to a facet have a non-zero normal trace on it. The cell
/// chosen is the one with the larger global index.
///
/// This asks nothing of the surface as a whole, only of each facet in
/// turn, so it works on a surface that cannot be oriented, such as a
/// Möbius strip, as well as on one that can. It is local to a process
/// but for one reduction.
///
/// @note Which cell of a facet carries the sign, and how many facets
/// are signed, depend on the partitioning, since both follow from the
/// global numbering. The space they define does not.
///
/// @pre The edges, the connectivity between edges and cells, and the
/// edge permutations must have been created.
/// @param[in] topology Topology of a mesh of triangles or
/// quadrilaterals.
/// @return For each owned and ghost cell and each of its facets, in
/// cell-local facet order and flattened row-wise, `1` if the
/// degrees-of-freedom on that facet must be negated and `0` otherwise.
/// @note Collective.
/// @throws std::invalid_argument If the topological dimension is not 2.
/// @throws std::runtime_error If an edge is shared by more than two
/// cells, or if a facet needs correcting on a process that holds no
/// ghost cells and so cannot see the cell on its other side.
std::vector<std::uint8_t> compute_facet_orientations(const Topology& topology);

} // namespace dolfinx::mesh
