// Copyright (C) 2020-2022 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/mesh/cell_types.h>
#include <tuple>
#include <vector>
#include <xtensor/xtensor.hpp>

namespace dolfinx
{
namespace fem
{
class FunctionSpace;
}

namespace mesh
{
enum class CellType;
class Mesh;
} // namespace mesh

namespace io
{
/// Given a FunctionSpace, create a topology and geometry based on the
/// dof coordinates
/// @pre `V` must be a (discontinuous) Lagrange space
/// @param[in] V The function space
/// @returns Mesh data
/// -# node coordinates (shape={num_nodes, 3})
/// -# unique global ID for each node (a node that appears on more than
/// one rank will have the same global ID)
/// -# ghost index for each node (0=non-ghost, 1=ghost)
/// -# cells (shape={num_cells, nodes_per_cell)})
std::tuple<xt::xtensor<double, 2>, std::vector<std::int64_t>,
           std::vector<std::uint8_t>, xt::xtensor<std::int32_t, 2>>
vtk_mesh_from_space(const fem::FunctionSpace& V);

/// Extract the cell topology (connectivity) in VTK ordering for all
/// cells the mesh. The 'topology' includes higher-order 'nodes'. The
/// index of a 'node' corresponds to the index of DOLFINx geometry
/// 'nodes'.
/// @param [in] mesh The mesh
/// @return The cell topology in VTK ordering and in term of the DOLFINx
/// geometry 'nodes'
/// @note The indices in the return array correspond to the point
/// indices in the mesh geometry array
/// @note Even if the indices are local (int32), both Fides and VTX
/// require int64 as local input
xt::xtensor<std::int64_t, 2> extract_vtk_connectivity(const mesh::Mesh& mesh);

} // namespace io
} // namespace dolfinx
