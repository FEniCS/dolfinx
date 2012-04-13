// Copyright (C) 2012 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2012-04-13
// Last changed: 2012-04-13

#include <sstream>

#include <dolfin/common/utils.h>
#include <dolfin/log/log.h>
#include "CSGOperators.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// CSGUnion
//-----------------------------------------------------------------------------
CSGUnion::CSGUnion(boost::shared_ptr<CSGGeometry> g0,
                   boost::shared_ptr<CSGGeometry> g1)
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
std::string CSGUnion::str(bool verbose) const
{
  assert(_g0);
  assert(_g1);

  std::stringstream s;

  if (verbose)
  {
    s << "<Union>\n"
      << "{\n"
      << indent(_g0->str(true))
      << "\n"
      << indent(_g1->str(true))
      << "\n}";
  }
  else
  {
    s << "(" << _g0->str(false) << " + " << _g1->str(false) << ")";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
// CSGIntersection
//-----------------------------------------------------------------------------
CSGIntersection::CSGIntersection(boost::shared_ptr<CSGGeometry> g0,
                                 boost::shared_ptr<CSGGeometry> g1)
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
std::string CSGIntersection::str(bool verbose) const
{
  assert(_g0);
  assert(_g1);

  std::stringstream s;

  if (verbose)
  {
    s << "<Intersection>\n"
      << "{\n"
      << indent(_g0->str(true))
      << "\n"
      << indent(_g1->str(true))
      << "\n}";
  }
  else
  {
    s << "(" << _g0->str(false) << " * " << _g1->str(false) << ")";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
