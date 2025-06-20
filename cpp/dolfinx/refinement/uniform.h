#include <dolfinx/mesh/Mesh.h>

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
/// @returns Uniformly refined mesh
template <typename T>
mesh::Mesh<T> uniform_refine(const mesh::Mesh<T>& mesh);

} // namespace dolfinx::refinement
