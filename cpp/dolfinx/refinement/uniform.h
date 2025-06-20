#include <dolfinx/mesh/Mesh.h>

#pragma once

namespace dolfinx::refinement
{

/// @brief Uniform refinement of a 2D or 3D mesh, containing any supported cell
/// types.
/// @param mesh Input mesh
/// @returns Uniformly refined mesh
mesh::Mesh<double> uniform_refine(const mesh::Mesh<double>& mesh);

} // namespace dolfinx::refinement
