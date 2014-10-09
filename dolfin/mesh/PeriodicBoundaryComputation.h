// Copyright (C) 2013 Garth N. Wells
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
// First added:  2012-01-10
// Last changed:

#ifndef __PERIODIC_BOUNDARY_COMPUTATION_H
#define __PERIODIC_BOUNDARY_COMPUTATION_H

#include <map>
#include <utility>
#include <vector>
#include <dolfin/common/constants.h>
#include <dolfin/mesh/MeshFunction.h>

namespace dolfin
{

  class Mesh;
  class SubDomain;

  /// This class computes map from slave entity to master entity

  class PeriodicBoundaryComputation
  {
  public:

    /// For entities of dimension dim, compute map from a slave entity
    /// on this process (local index) to its master entity (owning
    /// process, local index on owner). If a master entity is shared
    /// by processes, only one of the owning processes is returned.
    static std::map<unsigned int, std::pair<unsigned int, unsigned int> >
      compute_periodic_pairs(const Mesh& mesh, const SubDomain& sub_domain,
                             const std::size_t dim);

    /// This function returns a MeshFunction which marks mesh entities
    /// of dimension dim according to:
    ///
    ///     2: slave entities
    ///     1: master entities
    ///     0: all other entities
    ///
    /// It is useful for visualising and debugging the Expression::map
    /// function that is used to apply periodic boundary conditions.
    static MeshFunction<std::size_t>
      masters_slaves(std::shared_ptr<const Mesh> mesh,
                     const SubDomain& sub_domain, const std::size_t dim);

  private:

    // Return true is point lies within bounding box
    static bool in_bounding_box(const std::vector<double>& point,
                                const std::vector<double>& bounding_box,
                                const double tol);

  };

}

#endif
