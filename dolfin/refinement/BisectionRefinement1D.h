// Copyright (C) 2014 Chris Richardson
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
// Last changed: 2011-02-09

#ifndef __BISECTION_REFINEMENT_1D_H
#define __BISECTION_REFINEMENT_1D_H

namespace dolfin
{
  class Mesh;
  template<typename T> class MeshFunction;

  /// This class implements mesh refinement in 1D

  class BisectionRefinement1D
  {
  public:

    /// Refine mesh based on cell markers
    static void refine(Mesh& refined_mesh,
                       const Mesh& mesh,
                       const MeshFunction<bool>& cell_markers,
                       bool redistribute=false);

    /// Refine mesh uniformly
    static void refine(Mesh& refined_mesh,
                       const Mesh& mesh,
                       bool redistribute=false);

  };

}

#endif
