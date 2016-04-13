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

#ifndef __UNITHEXMESH_MESH_H
#define __UNITHEXMESH_MESH_H

#include <string>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

  /// NB: this code is experimental, just for testing, and will generally not
  /// work with anything else
  class UnitHexMesh : public Mesh
  {
  public:

    /// NB: this code is experimental, just for testing, and will generally not
    /// work with anything else
    UnitHexMesh(MPI_Comm comm, std::size_t nx, std::size_t ny, std::size_t nz);

  };

}

#endif
