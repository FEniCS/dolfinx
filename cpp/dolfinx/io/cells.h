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

/// Permutation array to map from VTK to DOLFINX node ordering
///
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return Permutation array @p for permuting from VTK ordering to
///   DOLFIN ordering, i.e. `a_dolfin[i] = a_vtk[p[i]]
/// @details If `p = [0, 2, 1, 3]` and `a = [10, 3, 4, 7]`, then `a_p =[a[p[0]],
///   a[p[1]], a[p[2]], a[p[3]]] = [10, 4, 3, 7]`
std::vector<std::uint8_t> perm_vtk(mesh::CellType type, int num_nodes);

/// Compute the transpose of a re-ordering map
///
/// @param[in] map A re-ordering map
/// @return Transpose of the @p map. E.g., is `map = {1, 2, 3, 0}`, the
///   transpose will be `{3 , 0, 1, 2 }`.
std::vector<std::uint8_t> transpose(const std::vector<std::uint8_t>& map);

/// Permute cell topology by applying a permutation array for each cell
/// @param[in] cells Array of cell topologies, with each row
///   representing a cell
/// @param[in] p The permutation array that maps `a_p[i] = a[p[i]]`,
///   where `a_p` is the permuted array
/// @return Permuted cell topology, where for a cell `v_new[i] =
///   v_old[map[i]]`
Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
compute_permutation(
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    const std::vector<std::uint8_t>& p);

} // namespace dolfinx::io::cells
