// Copyright (C) 2020-2022 Garth N. Wells and and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/mesh/cell_types.h>
#include <xtensor/xtensor.hpp>

namespace dolfinx::fem
{
class FunctionSpace;
}

namespace dolfinx::mesh
{
enum class CellType;
class Mesh;
} // namespace dolfinx::mesh

namespace dolfinx::io
{
/// Given a FunctionSpace, create a topology and geometry based on the
/// dof coordinates.
/// @note Only supports (discontinuous) Lagrange functions
/// @param[in] u The function
std::pair<xt::xtensor<double, 2>, xt::xtensor<std::int64_t, 2>>
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

} // namespace dolfinx::io
