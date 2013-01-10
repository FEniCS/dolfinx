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

#ifndef __PERIODIC_DOMAIN_H
#define __PERIODIC_DOMAIN_H

#include <map>
#include <utility>
#include <vector>

namespace dolfin
{

  class Mesh;
  class SubDomain;

  /// This class computes map from slave facet to master facet

  class PeriodicDomain
  {
  public:

    /// Compute map from a slave facet on this process (local index)
    /// to its master facet (owning process, local index on owner)
    static std::map<std::size_t, std::pair<std::size_t, std::size_t> >
      compute_periodic_facet_pairs(const Mesh& mesh,
                                   const SubDomain& sub_domain);

  private:

    // Return true is point lies within bounding box
    static bool in_bounding_box(const std::vector<double>& point,
                                const std::vector<double>& bounding_box);

  };

}

#endif
