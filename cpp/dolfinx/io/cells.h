// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <dolfinx/mesh/cell_types.h>
#include <vector>

namespace dolfinx
{
namespace io
{

namespace cells
{
/// For simplices the FEniCS ordering follows the UFC convention, see:
/// https://fossies.org/linux/ufc/doc/manual/ufc-user-manual.pdf For
/// non-simplices (quadrilaterals and hexahedrons) a tensor product
/// ordering, as specified in FIAT, is used.

/// Map from VTK node indices to DOLFINX node indicies
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return Map from local VTK index to the DOLFINX local index, i.e.
/// map[i] is the position of the ith VTK index in the DOLFINX ordering
std::vector<std::uint8_t> vtk_to_dolfin(mesh::CellType type, int num_nodes);

/// Map from the mapping of lexicographic nodes to a tensor product
/// ordering. Convert Lexicographic cell ordering to FEniCS cell
/// ordering. For simplices the FEniCS ordering is following the
/// UFC-convention, see:
/// https://fossies.org/linux/ufc/doc/manual/ufc-user-manual.pdf For
/// non-simplices (quadrilaterals and hexahedra a TensorProduct
/// ordering, as specified in FIAT. Lexicographical ordering is defined
/// as:
///
///   *--*--*   6--7--8  y
///   |     |   |     |  ^
///   *  *  *   3  4  5  |
///             |     |   |     |  |
///   *--*--*   0--1--2  ---> x
///
/// Tensor product:
///
///   *--*--*   1--7--4  y
///   |     |   |     |  ^
///   *  *  *   2  8  5  |
///             |     |   |     |  |
///   *--*--*   0--6--3  ---> x
///
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return The map
// std::vector<std::uint8_t> lex_to_tp(mesh::CellType type, int num_nodes);

/// Map from DOLFINX local indices to VTK local indices. It is the
/// transpose of vtk_to_dolfin
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return Map from local DOLFINX index to the VTK local index, i.e.
/// map[i] is the position of the ith DOLFNX index in the VTK ordering
std::vector<std::uint8_t> dolfin_to_vtk(mesh::CellType type, int num_nodes);

/// Re-order a collection of cell connections by applying a permutation
/// array
/// @param[in] cells Array of cell topologies, with each row
///     representing a cell
/// @param[in] permutation The permutation array to map to
/// @return Permted cell topology, where for a cell
///     v_new[permutation[i]] = v_old[i]
Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
permute_ordering(
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    const std::vector<std::uint8_t>& permutation);


} // namespace cells
} // namespace io
} // namespace dolfinx
