// Copyright (C) 2012 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2012-04-13
// Last changed: 2012-04-13

#include <dolfin/log/log.h>
#include "CSGOperators.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// CSGUnion
//-----------------------------------------------------------------------------
CSGUnion::CSGUnion(boost::shared_ptr<const CSGGeometry> g0,
                   boost::shared_ptr<const CSGGeometry> g1)
  : _g0(g0), _g1(g1)
{
  assert(g0);
  assert(g1);

  // Check dimensions
  if (g0->dim() != g1->dim())
  {
    dolfin_error("CSGOperators.cpp",
                 "create union of CSG geometries",
                 "Dimensions of geomestries don't match (%d vs %d)",
                 g0->dim(), g1->dim());
  }

}
//-----------------------------------------------------------------------------
dolfin::uint CSGUnion::dim() const
{
  assert(_g0->dim() == _g1->dim());
  return _g0->dim();
}
//-----------------------------------------------------------------------------
// CSGIntersection
//-----------------------------------------------------------------------------
CSGIntersection::CSGIntersection(boost::shared_ptr<const CSGGeometry> g0,
                                 boost::shared_ptr<const CSGGeometry> g1)
  : _g0(g0), _g1(g1)
{
  assert(g0);
  assert(g1);

  // Check dimensions
  if (g0->dim() != g1->dim())
  {
    dolfin_error("CSGOperators.cpp",
                 "create intersection of CSG geometries",
                 "Dimensions of geomestries don't match (%d vs %d)",
                 g0->dim(), g1->dim());
  }

}
//-----------------------------------------------------------------------------
dolfin::uint CSGIntersection::dim() const
{
  assert(_g0->dim() == _g1->dim());
  return _g0->dim();
}
//-----------------------------------------------------------------------------
