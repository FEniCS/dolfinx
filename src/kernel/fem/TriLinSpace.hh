// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TRILIN_SPACE_HH
#define __TRILIN_SPACE_HH

#include "FunctionSpace.hh"

class TriLinFunction;
class LocalField;

namespace dolfin {

  class TriLinSpace: public FunctionSpace {
  public:
	 
	 TriLinSpace(FiniteElement *element, int nvc);
	 ~TriLinSpace();
	 
	 void Update();
	 
	 friend class TriLinFunction;
	 friend class FiniteElement;
	 
  private:
	 
	 real IntShapeFunction(int m1);
	 real IntShapeFunction(int m1, int m2);
	 real IntShapeFunction(int m1, int m2, int m3);
	 
	 real g1x, g1y, g2x, g2y, g3x, g3y;
	 real det,d;
	 
  };

}

#endif
