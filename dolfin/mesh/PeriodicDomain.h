// Copyright (C) 2007-2008 Anders Logg
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
// Modified by Garth N. Wells 2007-2012
// Modified by Johan Hake 2009
//
// First added:  2007-07-08
// Last changed: 2012-08-18

#ifndef __PERIODIC_DOMAIN_H
#define __PERIODIC_DOMAIN_H

#include <map>
#include <utility>
#include <vector>

namespace dolfin
{

  class Mesh;
  class SubDomain;

  /// This class specifies the interface for setting periodic boundaries
  ///
  /// .. math::
  ///
  ///     u(x) &= u(F^{-1}(x)) \hbox { on } G,
  ///
  ///     u(x) &= u(F(x))      \hbox{ on } H,
  ///
  /// where F : H --> G is a map from a subdomain H to a subdomain G.
  ///
  /// A periodic boundary condition must be defined by the domain G
  /// and the map F pulling coordinates back from H to G. The domain
  /// and the map are both defined by a subclass of _SubDomain_ which
  /// must overload both the inside() function, which specifies the
  /// points of G, and the map() function, which specifies the map
  /// from the points of H to the points of G.
  ///
  /// The implementation is based on matching degrees of freedom on G
  /// with degrees of freedom on H and only works when the mapping F
  /// is bijective between the sets of coordinates associated with the
  /// two domains. In other words, the nodes (degrees of freedom) must
  /// be aligned on G and H.
  ///
  /// The matching of degrees of freedom is done at the construction
  /// of the periodic boundary condition and is reused on subsequent
  /// applications to a linear system. The matching may be recomputed
  /// by calling the ``rebuild()`` function.

  class PeriodicDomain
  {
  public:

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
