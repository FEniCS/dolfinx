// Copyright (C) 2010 Garth N. Wells
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
// Modified by Anders Logg, 2010.
//
// First added:  2010-02-10
// Last changed: 2010-02-26
//
// This file defines free functions for mesh refinement.
//

#ifndef __DOLFIN_REFINE_H
#define __DOLFIN_REFINE_H

namespace dolfin
{

  // Forward declarations
  class Mesh;
  template <class T> class MeshFunction;

  /// Create uniformly refined mesh
  Mesh refine(const Mesh& mesh);

  /// Create uniformly refined mesh
  void refine(Mesh& refined_mesh,
              const Mesh& mesh);

  /// Create locally refined mesh
  Mesh refine(const Mesh& mesh,
              const MeshFunction<bool>& cell_markers);

  /// Create locally refined mesh
  void refine(Mesh& refined_mesh,
              const Mesh& mesh,
              const MeshFunction<bool>& cell_markers);

}

#endif
