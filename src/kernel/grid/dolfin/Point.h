// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __POINT_H
#define __POINT_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>

namespace dolfin {
  
  class Point {
  public:
	 
	 Point();
	 Point(real x);
	 Point(real x, real y);
	 Point(real x, real y, real z);
	 
	 real dist(Point p);

	 Point midpoint(Point p);
	 
	 real x;
	 real y;
	 real z;

	 friend LogStream& operator<<(LogStream& stream, const Point& p);
	 
  };

}
  
#endif
