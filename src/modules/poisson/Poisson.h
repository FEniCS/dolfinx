// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __POISSON_H
#define __POISSON_H

#include <Equation.h>

namespace dolfin {

  class Poisson : public Equation {
  public:
	 
	 Poisson(Function &source): f(source) {}
	 
	 real lhs(ShapeFunction &u, ShapeFunction &v) {
		return u.dx * v.dx + u.dy * v.dy + u.dz * v.dz;


		dx(u) * dx(v) + dy(u) * dy(v) + dz(u)*dz(v);

		grad(u) * grad(v)
	 }
	 
	 real rhs(ShapeFunction &v) {
		return f * v;
	 }
	 
  private:
	 
	 ElementFunction &f;
	 
  };

}
  
#endif
