#include <dolfinx/mesh/Mesh.h>

#pragma once

namespace dolfinx::refinement
{

/// @brief Uniform refinement of a 3D mesh containing hex, tet, prism or pyramids.
/// @param mesh Input mesh
mesh::Mesh<double> uniform_refine(const mesh::Mesh<double>& mesh);

} // namespace dolfinx::refinement
