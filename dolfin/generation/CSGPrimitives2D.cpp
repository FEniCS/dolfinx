// Copyright (C) 2012 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2012-04-12
// Last changed: 2012-04-12

#include "CSGPrimitives2D.h"

using namespace dolfin;
using namespace dolfin::csg;

//-----------------------------------------------------------------------------
// Circle
//-----------------------------------------------------------------------------
Circle::Circle(double x0, double x1, double r)
  : _x0(x0), _x1(x1), _r(r)
{


}
//-----------------------------------------------------------------------------
// Rectangle
//-----------------------------------------------------------------------------
Rectangle::Rectangle(double x0, double x1, double y0, double y1)
  : _x0(x0), _x1(x1), _y0(y0), _y1(y1)
{


}
//-----------------------------------------------------------------------------
