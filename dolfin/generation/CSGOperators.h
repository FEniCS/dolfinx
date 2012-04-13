// Copyright (C) 2012 Anders Logg
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
// First added:  2012-04-13
// Last changed: 2012-04-13

#ifndef __CSG_OPERATORS_H
#define __CSG_OPERATORS_H

#include <boost/shared_ptr.hpp>

#include <dolfin/common/NoDeleter.h>
#include "CSGGeometry.h"

namespace dolfin
{

  //--- Operator classes (nodes) ---

  /// Union of CSG geometries
  class CSGUnion : public CSGGeometry
  {
  public:

    /// Create union of two geometries
    CSGUnion(boost::shared_ptr<CSGGeometry> g0,
             boost::shared_ptr<CSGGeometry> g1);

    /// Return dimension of geometry
    uint dim() const;

    /// Informal string representation
    std::string str(bool verbose) const;

  private:

    boost::shared_ptr<CSGGeometry> _g0;
    boost::shared_ptr<CSGGeometry> _g1;

  };

  /// Intersection of CSG geometries
  class CSGIntersection : public CSGGeometry
  {
  public:

    /// Create intersection of two geometries
    CSGIntersection(boost::shared_ptr<CSGGeometry> g0,
                    boost::shared_ptr<CSGGeometry> g1);

    /// Return dimension of geometry
    uint dim() const;

    /// Informal string representation
    std::string str(bool verbose) const;

  private:

    boost::shared_ptr<CSGGeometry> _g0;
    boost::shared_ptr<CSGGeometry> _g1;

  };

  //--- Union operators ---

  /// Create union of two geometries
  boost::shared_ptr<CSGUnion> operator+(boost::shared_ptr<CSGGeometry> g0,
                                        boost::shared_ptr<CSGGeometry> g1)
  {
    return boost::shared_ptr<CSGUnion>(new CSGUnion(g0, g1));
  }

  /// Create union of two geometries
  boost::shared_ptr<CSGUnion> operator+(CSGGeometry& g0,
                                        boost::shared_ptr<CSGGeometry> g1)
  {
    return reference_to_no_delete_pointer(g0) + g1;
  }

  /// Create union of two geometries
  boost::shared_ptr<CSGUnion> operator+(boost::shared_ptr<CSGGeometry> g0,
                                        CSGGeometry& g1)
  {
    return g0 + reference_to_no_delete_pointer(g1);
  }

  /// Create union of two geometries
  boost::shared_ptr<CSGUnion> operator+(CSGGeometry& g0,
                                        CSGGeometry& g1)
  {
    return reference_to_no_delete_pointer(g0) + reference_to_no_delete_pointer(g1);
  }

  //--- Intersection operators ---

  /// Create intersection of two geometries
  boost::shared_ptr<CSGIntersection> operator*(boost::shared_ptr<CSGGeometry> g0,
                                               boost::shared_ptr<CSGGeometry> g1)
  {
    return boost::shared_ptr<CSGIntersection>(new CSGIntersection(g0, g1));
  }

  /// Create intersection of two geometries
  boost::shared_ptr<CSGIntersection> operator*(CSGGeometry& g0,
                                               boost::shared_ptr<CSGGeometry> g1)
  {
    return reference_to_no_delete_pointer(g0) * g1;
  }

  /// Create intersection of two geometries
  boost::shared_ptr<CSGIntersection> operator*(boost::shared_ptr<CSGGeometry> g0,
                                               CSGGeometry& g1)
  {
    return g0 * reference_to_no_delete_pointer(g1);
  }

  /// Create intersection of two geometries
  boost::shared_ptr<CSGIntersection> operator*(CSGGeometry& g0,
                                               CSGGeometry& g1)
  {
    return reference_to_no_delete_pointer(g0) * reference_to_no_delete_pointer(g1);
  }

}

#endif
