// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// A simple structure containing the coordinates of a node. This
// is where a large part of the grid is physically stored so it
// should be small and simple.

#ifndef __POINT_H
#define __POINT_H

#include <iostream>
#include <dolfin/constants.h>

namespace dolfin {
  
  class Point {
  public:
	 
	 Point();
	 Point(real x);
	 Point(real x, real y);
	 Point(real x, real y, real z);
	 
	 real dist(Point p);
	 
	 real x;
	 real y;
	 real z;

	 friend std::ostream& operator << (std::ostream& output, const Point& p);
	 
  };

}
  
#endif
