// Copyright (C) 2006-2008 Anders Logg
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
// Modified by Niclas Jansson 2009.
//
// First added:  2006-06-21
// Last changed: 2010-03-02

#ifndef __BOUNDARY_COMPUTATION_H
#define __BOUNDARY_COMPUTATION_H

#include <vector>

namespace dolfin
{

  class BoundaryMesh;
  class Facet;
  class Mesh;
  template <typename T> class MeshFunction;

  /// This class implements provides a set of basic algorithms for
  /// the computation of boundaries.

  class BoundaryComputation
  {
  public:

    /// Compute the exterior boundary of a given mesh
    static void compute_exterior_boundary(const Mesh& mesh,
                                          BoundaryMesh& boundary);

    /// Compute the interior boundary of a given mesh
    static void compute_interior_boundary(const Mesh& mesh,
                                          BoundaryMesh& boundary);

  private:

    /// Compute the boundary of a given mesh
    static void compute_boundary_common(const Mesh& mesh,
					                              BoundaryMesh& boundary,
					                              bool interior_boundary);

    /// Reorder vertices so facet is right-oriented w.r.t. facet normal
    static void reorder(std::vector<std::size_t>& vertices, const Facet& facet);

  };

}

#endif
