// Copyright (C) 2012 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2012-04-12
// Last changed: 2012-04-12

#include "CSGPrimitives3D.h"

using namespace dolfin;
using namespace dolfin::csg;

//-----------------------------------------------------------------------------
// Sphere
//-----------------------------------------------------------------------------
Sphere::Sphere(double x0, double x1, double x2, double r)
  : _x0(x0), _x1(x1), _x2(x2), _r(r)
{


}
//-----------------------------------------------------------------------------
// Box
//-----------------------------------------------------------------------------
Box::Box(double x0, double x1, double x2,
         double y0, double y1, double y2)
  : _x0(x0), _x1(x1), _x2(x2), _y0(y0), _y1(y1), _y2(y2)
{


}
//-----------------------------------------------------------------------------
