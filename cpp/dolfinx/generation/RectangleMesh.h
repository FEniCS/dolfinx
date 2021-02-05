// Copyright (C) 2005-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Core>
#include <array>
#include <dolfinx/common/MPI.h>
#include <dolfinx/mesh/Mesh.h>
#include <string>

namespace dolfinx
{
namespace fem
{
class CoordinateElement;
}

/// Rectangle mesh creation
namespace generation::RectangleMesh
{

/// Create a uniform mesh::Mesh over the rectangle spanned by the two
/// points @p p. The order of the two points is not important in terms
/// of minimum and maximum coordinates. The total number of vertices
/// will be `(n[0] + 1)*(n[1] + 1)`. For triangles there will be  will
/// be `2*n[0]*n[1]` cells. For quadrilaterals the number of cells
/// will be `n[0]*n[1]`.
///
/// @param[in] comm MPI communicator to build the mesh on
/// @param[in] p Two corner points
/// @param[in] n Number of cells in each direction
/// @param[in] element Element that describes the geometry of a cell
/// @param[in] ghost_mode Mesh ghosting mode
/// @param[in] diagonal Direction of diagonals: "left", "right",
/// "left/right", "crossed"
/// @return Mesh
mesh::Mesh create(MPI_Comm comm, const std::array<Eigen::Vector3d, 2>& p,
                  std::array<std::size_t, 2> n,
                  const fem::CoordinateElement& element,
                  const mesh::GhostMode ghost_mode,
                  const std::string& diagonal = "right");

/// Create a uniform mesh::Mesh over the rectangle spanned by the two
/// points @p p. The order of the two points is not important in terms
/// of minimum and maximum coordinates. The total number of vertices
/// will be `(n[0] + 1)*(n[1] + 1)`. For triangles there will be  will
/// be `2*n[0]*n[1]` cells. For quadrilaterals the number of cells
/// will be `n[0]*n[1]`.
///
/// @param[in] comm MPI communicator to build the mesh on
/// @param[in] p Two corner points
/// @param[in] n Number of cells in each direction
/// @param[in] element Element that describes the geometry of a cell
/// @param[in] ghost_mode Mesh ghosting mode
/// @param[in] partitioner Partitioning function to use for
/// determining the parallel distribution of cells across MPI ranks
/// @param[in] diagonal Direction of diagonals: "left", "right",
/// "left/right", "crossed"
/// @return Mesh
mesh::Mesh create(MPI_Comm comm, const std::array<Eigen::Vector3d, 2>& p,
                  std::array<std::size_t, 2> n,
                  const fem::CoordinateElement& element,
                  const mesh::GhostMode ghost_mode,
                  const mesh::CellPartitionFunction& partitioner,
                  const std::string& diagonal = "right");
} // namespace generation::RectangleMesh
} // namespace dolfinx