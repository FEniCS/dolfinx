// Copyright (C) 2006-2010 Anders Logg
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
// Modified by Garth N. Wells, 2010
//
// First added:  2006-06-07
// Last changed: 2010-02-26

#ifndef __UNIFORM_MESH_REFINEMENT_H
#define __UNIFORM_MESH_REFINEMENT_H

#include <dolfin/common/types.h>

namespace dolfin
{

  class Mesh;

  /// This class implements uniform mesh refinement.

  class UniformMeshRefinement
  {
  public:

    /// Refine mesh uniformly
    static void refine(Mesh& refined_mesh, const Mesh& mesh);

  };

}

#endif
