#include <dolfinx/mesh/Mesh.h>

#pragma once

namespace dolfinx::refinement
{

/// @brief Uniform refinement of a 2D or 3D mesh, containing any supported cell
/// types.
/// @tparam T Scalar type of the mesh geometry
/// @param mesh Input mesh
/// @returns Uniformly refined mesh
template <typename T>
mesh::Mesh<T> uniform_refine(const mesh::Mesh<T>& mesh);

} // namespace dolfinx::refinement
