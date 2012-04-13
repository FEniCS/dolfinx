// Copyright (C) 2012 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
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
    CSGUnion(boost::shared_ptr<const CSGGeometry> g0,
             boost::shared_ptr<const CSGGeometry> g1);

    /// Return dimension of geometry
    uint dim() const;

  private:

    boost::shared_ptr<const CSGGeometry> _g0;
    boost::shared_ptr<const CSGGeometry> _g1;

  };

  /// Intersection of CSG geometries
  class CSGIntersection : public CSGGeometry
  {
  public:

    /// Create intersection of two geometries
    CSGIntersection(boost::shared_ptr<const CSGGeometry> g0,
                    boost::shared_ptr<const CSGGeometry> g1);

    /// Return dimension of geometry
    uint dim() const;

  private:

    boost::shared_ptr<const CSGGeometry> _g0;
    boost::shared_ptr<const CSGGeometry> _g1;

  };

  //--- Union operators ---

  /// Create union of two geometries
  boost::shared_ptr<const CSGUnion> operator+(boost::shared_ptr<const CSGGeometry> g0,
                                              boost::shared_ptr<const CSGGeometry> g1)
  {
    return boost::shared_ptr<CSGUnion>(new CSGUnion(g0, g1));
  }

  /// Create union of two geometries
  boost::shared_ptr<const CSGUnion> operator+(const CSGGeometry& g0,
                                              boost::shared_ptr<const CSGGeometry> g1)
  {
    return reference_to_no_delete_pointer(g0) + g1;
  }

  /// Create union of two geometries
  boost::shared_ptr<const CSGUnion> operator+(boost::shared_ptr<const CSGGeometry> g0,
                                              const CSGGeometry& g1)
  {
    return g0 + reference_to_no_delete_pointer(g1);
  }

  /// Create union of two geometries
  boost::shared_ptr<const CSGUnion> operator+(const CSGGeometry& g0,
                                              const CSGGeometry& g1)
  {
    return reference_to_no_delete_pointer(g0) + reference_to_no_delete_pointer(g1);
  }

  //--- Intersection operators ---

  /// Create intersection of two geometries
  boost::shared_ptr<const CSGIntersection>
  operator*(boost::shared_ptr<const CSGGeometry> g0,
            boost::shared_ptr<const CSGGeometry> g1)
  {
    return boost::shared_ptr<CSGIntersection>(new CSGIntersection(g0, g1));
  }

  /// Create intersection of two geometries
  boost::shared_ptr<const CSGIntersection>
  operator*(const CSGGeometry& g0,
            boost::shared_ptr<const CSGGeometry> g1)
  {
    return reference_to_no_delete_pointer(g0) * g1;
  }

  /// Create intersection of two geometries
  boost::shared_ptr<const CSGIntersection>
  operator*(boost::shared_ptr<const CSGGeometry> g0,
            const CSGGeometry& g1)
  {
    return g0 * reference_to_no_delete_pointer(g1);
  }

  /// Create intersection of two geometries
  boost::shared_ptr<const CSGIntersection>
  operator*(const CSGGeometry& g0,
            const CSGGeometry& g1)
  {
    return reference_to_no_delete_pointer(g0) * reference_to_no_delete_pointer(g1);
  }

}

#endif
