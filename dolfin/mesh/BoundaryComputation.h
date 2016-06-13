// Copyright (C) 2006-2013 Anders Logg and Garth N. Wells
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

#ifndef __BOUNDARY_COMPUTATION_H
#define __BOUNDARY_COMPUTATION_H

#include <string>
#include <vector>

namespace dolfin
{

  class BoundaryMesh;
  class Facet;
  class Mesh;
  template <typename T> class MeshFunction;

  /// Provide a set of basic algorithms for the computation of boundaries.

  class BoundaryComputation
  {
  public:

    /// Compute the exterior boundary of a given mesh
    /// @param[in] mesh  input mesh
    /// @param[in] type "internal" or "external"
    /// @param[out] boundary  output boundary mesh
    static void compute_boundary(const Mesh& mesh, const std::string type,
                                 BoundaryMesh& boundary);

  private:

    // Reorder vertices so facet is right-oriented w.r.t. facet normal
    static void reorder(std::vector<std::size_t>& vertices, const Facet& facet);

  };

}

#endif
