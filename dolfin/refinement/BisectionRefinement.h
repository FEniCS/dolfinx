// Copyright (C) 2006 Johan Hoffman
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
// Modified by Garth N. Wells, 2010-2011.
// Modified by Anders Logg, 2010-2011.
//
// First added:  2006-11-01
// Last changed: 2011-03-12

#ifndef __BISECTION_REFINEMENT_H
#define __BISECTION_REFINEMENT_H

namespace dolfin
{
  // Forward declarations
  class Mesh;
  template<typename T> class MeshFunction;

  /// This class implements local mesh refinement by edge bisection.

  class BisectionRefinement
  {
  public:

    /// Recursively refine mesh locally by longest edge bisection
    /// (Rivara). Fast Rivara algorithm implementation with
    /// propagation of MeshFunctions and arrays for boundary
    /// indicators.
    static void refine_by_recursive_bisection(Mesh& refined_mesh,
                                              const Mesh& mesh,
                                              const MeshFunction<bool>& cell_marker);

  // private:

    /// Transform mesh data
    //static void transform_data(Mesh& newmesh, const Mesh& oldmesh,
    //                           const MeshFunction<uint>& cell_map,
    //                           const std::vector<int>& facet_map);

  };

}

#endif
