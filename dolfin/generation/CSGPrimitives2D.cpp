// Copyright (C) 2012 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2012-04-12
// Last changed: 2012-04-13

#include <sstream>

#include "CSGPrimitives2D.h"

using namespace dolfin;
using namespace dolfin::csg;

//-----------------------------------------------------------------------------
// Circle
//-----------------------------------------------------------------------------
Circle::Circle(double x0, double x1, double r)
  : _x0(x0), _x1(x1), _r(r)
{
  // FIXME: Check validity of coordinates here
}
//-----------------------------------------------------------------------------
std::string Circle::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "<Circle at (" << _x0 << ", " << _x1 << ") with radius " << _r << ">";
  }
  else
  {
    s << "Circle("
      << _x0 << ", " << _x1 << ", " << _r << ")";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
// Rectangle
//-----------------------------------------------------------------------------
Rectangle::Rectangle(double x0, double x1, double y0, double y1)
  : _x0(x0), _x1(x1), _y0(y0), _y1(y1)
{
  // FIXME: Check validity of coordinates here
}
//-----------------------------------------------------------------------------
std::string Rectangle::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "<Rectangle with first corner at (" << _x0 << ", " << _x1 << ") "
      << "and second corner at (" << _y0 << ", " << _y1 << ")>";
  }
  else
  {
    s << "Rectangle("
      << _x0 << ", " << _x1 << ", " << _y0 << ", " << _y1 << ")";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
