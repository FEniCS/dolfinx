// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EQUATION_SYSTEM_H
#define __EQUATION_SYSTEM_H

#include <dolfin/Equation.h>

namespace dolfin {

  class EquationSystem : public Equation {
  public:
	 
	 EquationSystem(int dim, int noeq);
	 
	 virtual real lhs(ShapeFunction *u, ShapeFunction *v) = 0;
	 virtual real rhs(ShapeFunction *v) = 0;
	 
	 //--- Implementation of integrators for base class

	 real lhs(const ShapeFunction &u, const ShapeFunction &v);	 
	 real rhs(const ShapeFunction &v);
	 
  };

}

#endif
