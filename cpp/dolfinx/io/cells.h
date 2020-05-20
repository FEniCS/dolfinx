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

/// Functions for the re-ordering of input mesh topology to the DOLFINX
/// ordering, and transpose orderings for file output.
namespace dolfinx::io::cells
{

// For simplices the FEniCS ordering follows the UFC convention, see:
// https://fossies.org/linux/ufc/doc/manual/ufc-user-manual.pdf For
// non-simplices (quadrilaterals and hexahedrons) a tensor product
// ordering, as specified in FIAT, is used.

/// Map from VTK node indices to DOLFINX node indicies
///
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return Map `p` from the position i in the VTK array to position
///   `p[i] = j` in the  DOLFINX array, i.e. `a_dolfin[p[i]] =
///   a_vtk[i]`.
/// @details If `p = [0, 2, 1, 3]` and `a = [10, 3, 4, 7]`, then `a_p =
///   [10, 4, 3, 7]`.
std::vector<std::uint8_t> vtk_to_dolfin(mesh::CellType type, int num_nodes);

/// Compute the transpose of a re-ordering map
///
/// @param[in] map A re-ordering map
/// @return Transpose of the @p map. E.g., is `map = {1, 2, 3, 0}`, the
///   transpose will be `{3 , 0, 1, 2 }`.
std::vector<std::uint8_t> transpose(const std::vector<std::uint8_t>& map);

/// Re-order a collection of cell topology by applying a re-mapping
/// array
/// @param[in] cells Array of cell topologies, with each row
///   representing a cell
/// @param[in] map The map from the index in the cell array @p cell to
///   the position in the reorderd cell array, i.e. cell to to from
/// @return Permted cell topology, where for a cell `v_new[map[i]] =
///   v_old[i]`
Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
compute_reordering(
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    const std::vector<std::uint8_t>& map);

} // namespace dolfinx::io::cells
