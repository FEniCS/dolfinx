#include "HDF5Interface.h"

namespace dolfinx::mesh
{
template <std::floating_point T>
class Mesh;
}

namespace dolfinx::io::VTKHDF
{

/// Write a mesh to VTKHDF format
/// @param filename
/// @param mesh
template <typename U>
void write_mesh(std::string filename, const mesh::Mesh<U>& mesh);

} // namespace dolfinx::io::VTKHDF
