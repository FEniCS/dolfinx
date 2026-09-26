// Copyright (C) 2020 Matthew Scroggs
// Copyright (C) 2020-2026 Matthew Scroggs and Jørgen S. Dokken
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

/// @brief Bit of the cell permutation info that marks a cell whose
/// orientation is reversed relative to the orientation of a manifold
/// mesh.
///
/// Set by Topology::create_cell_orientations. The sub-entity permutations
/// packed into the same integer (see compute_cell_permutations) use at
/// most 30 bits (hexahedron: three per face and one per edge), so this
/// bit is free for every cell type.
constexpr std::uint32_t reversed_cell_bit = std::uint32_t(1) << 31;

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
/// they are reversed. The most significant bit is left for
/// ::reversed_cell_bit.
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

/// @brief Compute a consistent orientation of the cells of a surface
/// mesh.
///
/// Two cells sharing an edge are consistently oriented when, going
/// round each cell in the order of its vertices, they run the edge in
/// opposite directions. Each connected part of the surface, i.e. each
/// set of cells connected through shared edges, is walked across its
/// edges from its cell with the lowest global index, which keeps
/// orientation `1`. Cells meeting only at a vertex are oriented
/// independently. The geometry is not used. Which of its two
/// orientations a part gets depends on the partitioning, but reversing
/// all cells of a part negates all of its basis functions, which leaves
/// every field unchanged.
///
/// @note Collective.
/// @pre The edges, the connectivity between edges and cells, and the
/// edge permutations (see Topology::create_entity_permutations) must
/// have been created.
/// @param[in] topology Topology of a mesh of triangles or
/// quadrilaterals.
/// @return For each owned and ghost cell, in local cell order, `1` if
/// its vertex order agrees with the orientation of its part of the
/// surface and `-1` otherwise.
/// @throws std::invalid_argument If the topological dimension is not
/// 2.
/// @throws std::runtime_error If the surface is not orientable, e.g. a
/// Möbius strip, or an edge is shared by more than two cells.
std::vector<std::int8_t> compute_cell_orientations(const Topology& topology);

} // namespace dolfinx::mesh
