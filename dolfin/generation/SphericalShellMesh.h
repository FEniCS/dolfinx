// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//

#ifndef __SPHERICAL_SHELL_MESH_H
#define __SPHERICAL_SHELL_MESH_H

#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

  /// Spherical shell approximation, icosahedral mesh, with degree=1 or degree=2

  class SphericalShellMesh : public Mesh
  {
  public:

    /// Create a spherical shell manifold for testing
    SphericalShellMesh(MPI_Comm comm, std::size_t degree);

  };

}

#endif
