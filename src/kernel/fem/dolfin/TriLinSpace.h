// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TRILIN_SPACE_H
#define __TRILIN_SPACE_H

#include <dolfin/FunctionSpace.h>
#include <dolfin/ShapeFunction.h>

namespace dolfin {
  
  class TriLinSpace : public FunctionSpace {
  public:

	 // Definition of the local function space
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

	 // Mapping from local to global degrees of freedom
	 int dof(int i, const Cell &cell) const {
		return cell.nodeID(i);
	 }

	 // Evalutation of degrees of freedom
	 real dof(int i, const Cell &cell, function f, real t) const {
		Point p = cell.coord(i);
		return f(p.x, p.y, p.z, t);
	 }

  };
  
}

#endif
