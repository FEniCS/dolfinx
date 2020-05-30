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
/*
  The FIAT ordering is used for the geometry nodes, and is shown below
  for a range of cell types.

    Triangle:               Triangle6:          Triangle10:
    v
    ^
    |
    2                       2                    2
    |`\                     |`\                  | \
    |  `\                   |  `\                6   4
    |    `\                 4    `3              |     \
    |      `\               |      `\            5   9   3
    |        `\             |        `\          |         \
    0----------1 --> u      0-----5----1         0---7---8---1

    Quadrilateral:         Quadrilateral9:         Quadrilateral16:
    v
    ^
    |
    1-----------3          1-----7-----4           1---9--13---5
    |           |          |           |           |           |
    |           |          |           |           3  11  15   7
    |           |          2     8     5           |           |
    |           |          |           |           2  10  14   6
    |           |          |           |           |           |
    0-----------2 --> u    0-----6-----3           0---8--12---4

    Tetrahedron:                    Tetrahedron10:               Tetrahedron20
                v
               /
              2                            2                         2
            ,/|`\                        ,/|`\                     ,/|`\
          ,/  |  `\                    ,/  |  `\                 13  |  `9
         ,/    '.   `\                ,8    '.   `6           ,/     4   `\
       ,/       |     `\            ,/       4     `\       12    19 |     `8
     ,/         |       `\        ,/         |       `\   ,/         |       `\
    0-----------'.--------1 -> u 0--------9--'.--------1 0-----14----'.--15----1
     `\.         |      ,/        `\.         |      ,/   `\.  17     |  16 ,/
        `\.      |    ,/             `\.      |    ,5       10.   18 5    ,6
           `\.   '. ,/                  `7.   '. ,/            `\.   '.  7
              `\. |/                       `\. |/                 11. |/
                 `3                            `3                    `3
                    `\.
                       w

    Hexahedron:          Hexahedron27:
           v
    2----------6           3----21----12
    |\     ^   |\          |\         |\
    | \    |   | \         | 5    23  | 14
    |  \   |   |  \        6  \ 24    15 \
    |   3------+---7       |   4----22+---13
    |   |  +-- |-- | -> u  | 8 |  26  | 17|
    0---+---\--4   |       0---+18----9   |
     \  |    \  \  |        \  7     25\ 16
      \ |     \  \ |         2 |   20   11|
       \|      w  \|          \|         \|
       1----------5           1----19----10
*/

/// Permutation array to map from VTK to DOLFINX node ordering
///
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return Permutation array @p for permuting from VTK ordering to
///   DOLFIN ordering, i.e. `a_dolfin[i] = a_vtk[p[i]]
/// @details If `p = [0, 2, 1, 3]` and `a = [10, 3, 4, 7]`, then `a_p =[a[p[0]],
///   a[p[1]], a[p[2]], a[p[3]]] = [10, 4, 3, 7]`
std::vector<std::uint8_t> perm_vtk(mesh::CellType type, int num_nodes);

/// Permutation array to map from Gmsh to DOLFINX node ordering
///
/// @param[in] type The cell shape
/// @param[in] num_nodes
/// @return Permutation array @p for permuting from Gmsh ordering to
///   DOLFIN ordering, i.e. `a_dolfin[i] = a_gmsh[p[i]]
/// @details If `p = [0, 2, 1, 3]` and `a = [10, 3, 4, 7]`, then `a_p
///   =[a[p[0]], a[p[1]], a[p[2]], a[p[3]]] = [10, 4, 3, 7]`
std::vector<std::uint8_t> perm_gmsh(mesh::CellType type, int num_nodes);

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
