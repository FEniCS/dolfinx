// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __POINT_HH
#define __POINT_HH

// A simple structure containing the coordinates of a node. This
// is where a large part of the grid is physically stored so it
// should be small and simple.
//
// Coordinates are represented as floats to save memory, whereas
// calculations on node positions are performed with full precision.

#include <kw_constants.h>

class Point{
public:

  Point(){
	 x = 0.0;
	 y = 0.0;
	 z = 0.0;
  }
  
  real Distance(Point p);
  
  float x;
  float y;
  float z;
};

#endif
