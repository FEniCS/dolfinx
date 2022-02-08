// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/mesh/cell_types.h>
#include <vector>
#include <xtensor/xtensor.hpp>

/// Functions for the re-ordering of input mesh topology to the DOLFINx
/// ordering, and transpose orderings for file output.
namespace dolfinx::io::cells
{
/*
  The basix ordering is used for the geometry nodes, and is shown below
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
    2-----------3          2-----7-----3           2--10--11---3
    |           |          |           |           |           |
    |           |          |           |           7  14  15   9
    |           |          5     8     6           |           |
    |           |          |           |           6  12  13   8
    |           |          |           |           |           |
    0-----------1 --> u    0-----4-----1           0---4---5---1

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
            w
        6----------7               6----19----7
       /|   ^   v /|              /|         /|
      / |   |  / / |            17 |  25    18|
     /  |   | / /  |            / 14    24 /  15
    4----------5   |           4----16----5   |
    |   |   +--|---|--> u      |22 |  26  | 23|
    |   2------+---3           |   2----13+---3
    |  /       |  /           10  / 21   12  /
    | /        | /             | 9    20  | 11
    |/         |/              |/         |/
    0----------1               0-----8----1

*/

/// Permutation array to map from VTK to DOLFINx node ordering
///
/// @param[in] type The cell shape
/// @param[in] num_nodes The number of cell 'nodes'
/// @return Permutation array @p for permuting from VTK ordering to
/// DOLFINx ordering, i.e. `a_dolfin[i] = a_vtk[p[i]]
/// @details If `p = [0, 2, 1, 3]` and `a = [10, 3, 4, 7]`, then `a_p
/// =[a[p[0]], a[p[1]], a[p[2]], a[p[3]]] = [10, 4, 3, 7]`
std::vector<std::uint8_t> perm_vtk(mesh::CellType type, int num_nodes);

/// Permutation array to map from Gmsh to DOLFINx node ordering
///
/// @param[in] type The cell shape
/// @param[in] num_nodes
/// @return Permutation array @p for permuting from Gmsh ordering to
/// DOLFINx ordering, i.e. `a_dolfin[i] = a_gmsh[p[i]]
/// @details If `p = [0, 2, 1, 3]` and `a = [10, 3, 4, 7]`, then `a_p
/// =[a[p[0]], a[p[1]], a[p[2]], a[p[3]]] = [10, 4, 3, 7]`
std::vector<std::uint8_t> perm_gmsh(mesh::CellType type, int num_nodes);

/// Compute the transpose of a re-ordering map
///
/// @param[in] map A re-ordering map
/// @return Transpose of the @p map. E.g., is `map = {1, 2, 3, 0}`, the
/// transpose will be `{3 , 0, 1, 2 }`.
std::vector<std::uint8_t> transpose(const std::vector<std::uint8_t>& map);

/// Permute cell topology by applying a permutation array for each cell
/// @param[in] cells Array of cell topologies, with each row
/// representing a cell
/// @param[in] p The permutation array that maps `a_p[i] = a[p[i]]`,
/// where `a_p` is the permuted array
/// @return Permuted cell topology, where for a cell `v_new[i] =
/// v_old[map[i]]`
xt::xtensor<std::int64_t, 2>
compute_permutation(const xt::xtensor<std::int64_t, 2>& cells,
                    const std::vector<std::uint8_t>& p);

/// Get VTK cell identifier
/// @param[in] cell The cell type
/// @param[in] dim The topological dimension of the cell
/// @return The VTK cell identifier
std::int8_t get_vtk_cell_type(mesh::CellType cell, int dim);

} // namespace dolfinx::io::cells
