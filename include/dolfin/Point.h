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

#include <dolfin/dolfin_constants.h>

class Point{
public:

  Point(){
	 x = 0.0;
	 y = 0.0;
	 z = 0.0;
  }
  
  real Distance(Point p);
  
  real x;
  real y;
  real z;
};

#endif
