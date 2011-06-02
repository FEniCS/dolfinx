// Copyright (C) 2011 Anders Logg
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
// First added:  2011-02-07
// Last changed: 2011-02-07

#ifndef __LOCAL_MESH_REFINEMENT_H
#define __LOCAL_MESH_REFINEMENT_H

namespace dolfin
{

  class Mesh;
  template<class T> class MeshFunction;

  /// This class is provides functionality for local (adaptive) mesh
  /// refinement. It is a wrapper for various algorithms for local
  /// mesh refinement implemented as part of DOLFIN and it delegates
  /// the refinement to a particular refinement algorithm based on the
  /// value of the global parameter "refinement_algorithm".

  class LocalMeshRefinement
  {
  public:

    /// Refine mesh based on cell markers
    static void refine(Mesh& refined_mesh,
                       const Mesh& mesh,
                       const MeshFunction<bool>& cell_markers);

  };

}

#endif
