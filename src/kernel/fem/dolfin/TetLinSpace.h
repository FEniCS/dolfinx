// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TETLIN_SPACE_H
#define __TETLIN_SPACE_H

#include <dolfin/FunctionSpace.h>
#include <dolfin/ShapeFunction.h>

namespace dolfin {

  class TetLinSpace : public FunctionSpace {
  public:

	 // Definition of the local function space
	 TetLinSpace() : FunctionSpace(4) {
		
		// Define shape functions
		ShapeFunction v0(tetlin0);
		ShapeFunction v1(tetlin1);
		ShapeFunction v2(tetlin2);
		ShapeFunction v3(tetlin3);

		// Add shape functions and specify derivatives
		add(v0, -1.0, -1.0, -1.0);
		add(v1,  1.0,  0.0,  0.0);
		add(v2,  0.0,  1.0,  0.0);
		add(v3,  0.0,  0.0,  1.0);
		
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
