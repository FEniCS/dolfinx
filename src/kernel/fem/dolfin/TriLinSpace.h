// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TRILIN_SPACE_HH
#define __TRILIN_SPACE_HH

#include <dolfin/FunctionSpace.h>

namespace dolfin {

  real trilin0 (real x, real y, real z, real t) { return 1 - x - y; }
  real trilin1 (real x, real y, real z, real t) { return x; }
  real trilin2 (real x, real y, real z, real t) { return y; }
  
  class TriLinSpace : public FunctionSpace {
  public:
	 
	 TriLinSpace() : FunctionSpace(3) {

		// Define shape functions
		ShapeFunction v0(trilin0);
		ShapeFunction v1(trilin1);
		ShapeFunction v2(trilin2);

		// Add shape functions and specify derivatives
		add(v0, -1.0, -1.0);
		add(v1,  1.0,  0.0);
		add(v2,  0.0,  1.0);

	 }
	 
  };
  
}

#endif
