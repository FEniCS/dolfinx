// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __POINT_H
#define __POINT_H

// A simple structure containing the coordinates of a node. This
// is where a large part of the grid is physically stored so it
// should be small and simple.
//
// Coordinates are represented as floats to save memory, whereas
// calculations on node positions are performed with full precision.

#include <dolfin/constants.h>

class Point{
public:

  Point();
  Point(real x);
  Point(real x, real y);
  Point(real x, real y, real z);
  
  real dist(Point p);
  
  real x;
  real y;
  real z;
  
};

#endif
