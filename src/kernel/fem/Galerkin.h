// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GALERKIN_H
#define __GALERKIN_H

namespace dolfin {

  class Galerkin {
  public:
	 
	 Galerkin();
	 ~Galerkin();
	 
	 // Assemble and set boundary conditions
	 void assemble(Grid &grid, Equation &equation, Matrix &A, Vector &b);
	 
	 // Assemble A and b individually, without setting boundary conditions
	 void assembleLHS(Grid &grid, Equation &equation, Matrix &A);
	 void assembleRHS(Grid &grid, Equation &equation, Vector &b);

	 // Set boundary conditions
	 void setBC(Grid &grid, Matrix &A, Vector &b);
  
  private:
	 
	 void alloc(Matrix &A);
	 void alloc(Vector &b);

	 int size;           // Number of unknowns
	 int noeq;           // Number of equations
	 int dim;            // Number of space dimensions
	 
	 dolfin_bc (*bc_function) (real x, real y, real z, int node, int component);
  };
  
#endif
