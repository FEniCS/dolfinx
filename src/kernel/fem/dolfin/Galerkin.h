// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Assembling from variational formulation. It is assumed that
// each degree of freedom is associated with a node. The id of
// the node is used as an index for the degrees of freedom.

#ifndef __GALERKIN_H
#define __GALERKIN_H

#include <dolfin/dolfin_elements.h>
#include <dolfin/dolfin_quadrature.h>
#include <dolfin/Mapping.h>
#include <dolfin/TriLinMapping.h>
#include <dolfin/TetLinMapping.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Vector.h>
#include <dolfin/SparseMatrix.h>
#include <dolfin/Equation.h>
#include <dolfin/EquationSystem.h>
#include <dolfin/Grid.h>

namespace dolfin {
  
  class Galerkin {
  public:

	 // Default constructor (default method will be used)
	 Galerkin();

	 // Constructor allowing specification of method
	 Galerkin(FiniteElement &element, Mapping &mapping, Quadrature &quadrature);

	 // Assemble and set boundary conditions
	 void assemble(Equation &equation, Grid &grid, Matrix &A, Vector &b);
	 
	 // Assemble A and b individually, without setting boundary conditions
	 void assembleLHS(Equation &equation, Grid &grid, Matrix &A);
	 void assembleRHS(Equation &equation, Grid &grid, Vector &b);

	 // Set boundary conditions
	 //void setBC(Grid &grid, Matrix &A, Vector &b);
  
  private:

	 void init(Grid &grid);
	 
	 void alloc(Matrix &A, Grid &grid);
	 void alloc(Vector &b, Grid &grid);

	 // Method data
	 FiniteElement* element;    // The finite element
	 Mapping*       mapping;    // Mapping from reference cell
	 Quadrature*    quadrature; // Quadrature on reference cell

	 // Data for default method
	 P1TriElement                p1TriElement;
	 P1TetElement                p1TetElement;
	 TriLinMapping               triLinMapping;
	 TetLinMapping               tetLinMapping;
	 TriangleVertexQuadrature    triangleVertexQuadrature;
	 TetrahedronVertexQuadrature tetrahedronVertexQuadrature;	 
	 
  };

}

#endif
