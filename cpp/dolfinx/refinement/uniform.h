#include <dolfinx/graph/partition.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/partition.h>

#pragma once

namespace dolfinx::refinement
{

/// @brief Uniform refinement of a 2D or 3D mesh, containing any supported cell
/// types.
/// Hexahedral, tetrahedral and prism cells are subdivided into 8, each being
/// similar to the original cell. Pyramid cells are subdivided into 5 similar
/// pyramids, plus 4 tetrahedra. Triangle and quadrilateral cells are subdivided
/// into 4 similar subcells.
/// @tparam T Scalar type of the mesh geometry
/// @param mesh Input mesh
/// @param partitioner Function to partition new mesh across processes.
/// @param ghost_mode Ghost mode of the refined mesh, passed to
/// `partitioner` if it is callable.
/// @returns Uniformly refined mesh
template <typename T>
mesh::Mesh<T>
uniform_refine(const mesh::Mesh<T>& mesh,
               const graph::partition_fn& partitioner = graph::partition_graph,
               mesh::GhostMode ghost_mode = mesh::GhostMode::none);

} // namespace dolfinx::refinement
