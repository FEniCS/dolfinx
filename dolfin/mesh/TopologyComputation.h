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
// Modified by Garth N. Wells 2012.
//
// First added:  2006-06-02
// Last changed: 2012-02-14

#ifndef __TOPOLOGY_COMPUTATION_H
#define __TOPOLOGY_COMPUTATION_H

#include <vector>

namespace dolfin
{

  class Mesh;

  /// This class implements a set of basic algorithms that automate
  /// the computation of mesh entities and connectivity.

  class TopologyComputation
  {
  public:

    /// Compute mesh entities of given topological dimension
    static std::size_t compute_entities(Mesh& mesh, std::size_t dim);

    /// Compute connectivity for given pair of topological dimensions
    static void compute_connectivity(Mesh& mesh, std::size_t d0,
                                     std::size_t d1);

  private:

    // Compute connectivity from transpose
    static void compute_from_transpose(Mesh& mesh, std::size_t d0,
                                       std::size_t d1);

    // Direct lookup of entity from vertices in a map
    static void compute_from_map(Mesh& mesh,
                                 std::size_t d0,
                                 std::size_t d1);

    // Compute connectivity from intersection
    static void compute_from_intersection(Mesh& mesh, std::size_t d0,
                                          std::size_t d1, std::size_t d);

  };

}

#endif
