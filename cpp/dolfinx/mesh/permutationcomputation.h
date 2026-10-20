// Copyright (C) 2020 Matthew Scroggs
// Copyright (C) 2020-2026 Matthew Scroggs, Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <vector>

namespace dolfinx::mesh
{
class Topology;

/// @brief Bit of the cell permutation info that negates the
/// degrees-of-freedom on facet `f` of a cell (see
/// Topology::create_facet_orientations).
///
/// A cell of topological dimension 2 has at most four facets, its edges,
/// in bits 27 to 30. Its sub-entity permutations (see
/// compute_cell_permutations) take one bit per edge, so the two do not
/// overlap. The bits are cleared (`info & ~orientation_mask(tdim)`)
/// before the info is passed to Basix.
///
/// @param[in] f Cell-local facet index, `0 <= f < 4`.
constexpr std::uint32_t reversed_facet_bit(int f)
{
  return std::uint32_t(1) << (27 + f);
}

/// @brief Bits of the cell permutation info that hold the facet signs of
/// a cell of topological dimension `tdim`.
///
/// Non-zero only for `tdim == 2`. A hexahedron's sub-entity permutations
/// already reach bit 29.
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
/// This measures the entities of dimension `dim`, which must therefore
/// exist, against the cells incident to them.
/// Topology::create_entity_permutations instead unpacks the same values
/// from the packed cell permutations, which need only the cell
/// vertices. The two are independent routes to the same data, and
/// `cpp/test/mesh/permutations.cpp` checks that they agree.
///
/// @pre The entities of dimension `dim`, and their connectivity with
/// the cells and the vertices, must already exist (see
/// Topology::create_entities).
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
/// @pre Cell-to-vertex connectivity and the vertex index map must exist.
/// @param[in] topology Mesh topology.
/// @param[in] num_threads Number of threads to use. Must be >= 1.
/// @return Facet permutation and cell permutations, covering both owned
/// and ghost cells.
/// @throws std::invalid_argument If `num_threads < 1`.
/// @throws std::runtime_error If `topology` is a mixed-topology mesh
/// (more than one 3D cell type).
std::vector<std::uint32_t> compute_cell_permutations(const Topology& topology,
                                                     int num_threads);

/// @brief Find the facets whose two cells disagree on the normal (see
/// Topology::create_facet_orientations).
///
/// Two cells disagree when they run their shared facet in the same
/// direction. The sign is then carried by the cell with the larger global
/// index. The cells of a facet on a process boundary are paired through
/// one exchange over the edge index map, so ghost cells are not needed
/// and are signed as by their owner.
///
/// @pre `topology.dim() == 2`, and the edges and the edge permutations
/// have been created.
/// @param[in] topology Topology of a mesh of triangles or
/// quadrilaterals.
/// @return For each owned and ghost cell, the mask of its signed
/// facets, with reversed_facet_bit(f) set if the degrees-of-freedom on
/// its facet `f` must be negated.
/// @note Collective.
/// @throws std::runtime_error If an edge is shared by more than two
/// cells, after all communication. On every process in Developer and
/// Debug builds, otherwise only on the processes holding such an edge.
std::vector<std::uint32_t> compute_facet_orientations(const Topology& topology);

} // namespace dolfinx::mesh
