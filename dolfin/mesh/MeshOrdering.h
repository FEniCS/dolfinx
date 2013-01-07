// Copyright (C) 2007-2008 Anders Logg
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
// First added:  2007-01-30
// Last changed: 2010-11-27

#ifndef __MESH_ORDERING_H
#define __MESH_ORDERING_H

namespace dolfin
{

  class Mesh;

  /// This class implements the ordering of mesh entities according to
  /// the UFC specification (see appendix of DOLFIN user manual).

  class MeshOrdering
  {
  public:

    /// Order mesh
    static void order(Mesh& mesh);

    /// Check if mesh is ordered
    static bool ordered(const Mesh& mesh);

  };

}

#endif
