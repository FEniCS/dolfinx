// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Mapping.h>
#include <dolfin/TriLinMapping.h>
#include <dolfin/TetLinMapping.h>
#include <dolfin/Quadrature.h>
#include <dolfin/TriangleVertexQuadrature.h>
//#include <dolfin/TetrahedronVertexQuadrature.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Galerkin.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Galerkin::Galerkin()
{
  // Will be created later (need to know the space dimension)
  element = 0;
  mapping = 0;
  quadrature = 0;
}
//-----------------------------------------------------------------------------
Galerkin::Galerkin(FiniteElement& element,
						 Mapping&       mapping,
						 Quadrature&    quadrature)
{
  this->element    = &element;
  this->mapping    = &mapping;
  this->quadrature = &quadrature;
}
//-----------------------------------------------------------------------------
void Galerkin::assemble(Equation &equation, Grid &grid, Matrix &A, Vector &b)
{
  assembleLHS(equation, grid, A);
  assembleRHS(equation, grid, b);
  //setBC(grid, A, b);
}
//-----------------------------------------------------------------------------
void Galerkin::assembleLHS(Equation &equation, Grid &grid, Matrix &A)
{  
  // Make sure that we have chosen trial and test spaces
  init(grid);
    
  // Allocate and reset matrix
  alloc(A, grid);

  // Write a message
  cout << "Assembling: system size is ";
  cout << A.size(0) << " x " << A.size(1) << "." << endl;
  
  // Iterate over all cells in the grid
  for (CellIterator cell(grid); !cell.end(); ++cell) {

	 // Update mapping
	 mapping->update(cell);

	 // Update equation
	 equation.updateLHS(element, cell, mapping, quadrature);
	 
	 // Iterate over test and trial functions
	 for (FiniteElement::TestFunctionIterator v(element); !v.end(); ++v)
		for (FiniteElement::TrialFunctionIterator u(element); !u.end(); ++u)
		  A(v.dof(cell), u.dof(cell)) += equation.lhs(u, v);
	 
  }
  
  // Clear unused elements
  A.resize();

  // Write a message
  cout << "Assembling: " << A << endl;
}
//-----------------------------------------------------------------------------
void Galerkin::assembleRHS(Equation &equation, Grid &grid, Vector &b)
{
  // Make sure that we have chosen trial and test spaces
  init(grid);
    
  // Allocate and reset matrix
  alloc(b, grid);

  // Iterate over all cells in the grid
  for (CellIterator cell(grid); !cell.end(); ++cell) {

	 	 // Update mapping
	 mapping->update(cell);

	 // Update equation
	 equation.updateRHS(element, cell, mapping, quadrature);
	 
	 // Iterate over test and trial functions
	 for (FiniteElement::TestFunctionIterator v(element); !v.end(); ++v)
		b(v.dof(cell)) += equation.rhs(v);
	 
  }
  
  // Write a message
  cout << "Assembling: " << b << endl;
}
//-----------------------------------------------------------------------------
/*
void Galerkin::setBC(Grid &grid, Matrix &A, Vector &b)
{

  if ( (A->Size(0) != no_eq*no_nodes ) || ( b->Size() != no_eq*no_nodes ) )
  	 display->Error("You must assemble the matrix before settings boundary conditions.");

  Point *p;
  dolfin_bc bc;
  
  for (int i=0;i<no_nodes;i++){

	 p  = grid->GetNode(i)->GetCoord();
	 
    for (int component=0;component<no_eq;component++){

      bc = bc_function(p->x,p->y,p->z,i,component+equation->GetStartVectorComponent());
      
      switch ( bc.type ){
      case dirichlet:
		  A->SetRowIdentity(i*no_eq+component);
		  b->Set(i*no_eq+component,bc.val);
		  break;
      case neumann:
		  if ( bc.val != 0.0 )
			 display->Error("Inhomogeneous Neumann boundary conditions not implemented.");
		  break;
      default:
		  display->InternalError("Galerkin::SetBoundaryConditions()",
										 "Unknown boundary condition type.");
      }
    }

  }


}
*/
//-----------------------------------------------------------------------------
void Galerkin::init(Grid &grid)
{
  // Check if the element has already been created
  if ( element )
	 return;

  // Create default finite element
  switch ( grid.type() ) {
  case Cell::TRIANGLE:
	 cout << "Using standard piecewise linears on triangles." << endl;
	 element    = new FiniteElement(triLinSpace, triLinSpace);
	 mapping    = new TriLinMapping();
	 quadrature = new TriangleVertexQuadrature();
	 break;
  case Cell::TETRAHEDRON:
	 cout << "Using standard piecewise linears on tetrahedrons." << endl;
	 element    = new FiniteElement(tetLinSpace, tetLinSpace);
	 mapping    = new TetLinMapping();
	 //quadrature = TetrahedronVertexQuadrature();
  break;
  default:
	 // FIXME: Use logging system
	 cout << "Error: No default spaces for this type of cells." << endl;
	 exit(1);
  }

}
//-----------------------------------------------------------------------------
void Galerkin::alloc(Matrix &A, Grid &grid)
{
  // Count the degrees of freedom
  
  int imax = 0;
  int jmax = 0;
  int i,j;

  for (CellIterator cell(grid); !cell.end(); ++cell) {

	 for (FiniteElement::TestFunctionIterator v(element); !v.end(); ++v)
		if ( (i = v.dof(cell)) > imax )
		  imax = i;

	 for (FiniteElement::TrialFunctionIterator u(element); !u.end(); ++u)
		if ( (j = u.dof(cell)) > jmax )
		  jmax = j;
		  
  }

  // Size of the matrix
  int m = imax + 1;
  int n = jmax + 1;
  
  // Initialise matrix
  if ( A.size(0) != m || A.size(1) != n )
	 A.init(m, n);

  // Set all entries to zero
  A = 0.0;
}
//-----------------------------------------------------------------------------
void Galerkin::alloc(Vector &b, Grid &grid)
{
  // Find number of degrees of freedom
  int imax = 0;
  int i;
  
  for (CellIterator cell(grid); !cell.end(); ++cell)
	 for (FiniteElement::TestFunctionIterator v(element); !v.end(); ++v)
		if ( (i = v.dof(cell)) > imax )
		  imax = i;

  // Size of vector
  int m = imax + 1;
  
  // Initialise vector
  if ( b.size() != m )
	 b.init(m);

  // Set all entries to zero
  b = 0.0;
}
//-----------------------------------------------------------------------------
