// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PDE_SYSTEM_H
#define __PDE_SYSTEM_H

#include <dolfin/PDE.h>

namespace dolfin {

  class PDESystem : public PDE {
  public:
	 
	 PDESystem(int dim, int noeq);
	 
	 virtual real lhs(ShapeFunction *u, ShapeFunction *v) = 0;
	 virtual real rhs(ShapeFunction *v) = 0;
	 
	 //--- Implementation of integrators for base class

	 real lhs(const ShapeFunction &u, const ShapeFunction &v);	 
	 real rhs(const ShapeFunction &v);
	 
  };

}

#endif
