// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_elements.h>
#include <dolfin/dolfin_quadrature.h>
#include <dolfin/Mapping.h>
#include <dolfin/TriLinMapping.h>
#include <dolfin/TetLinMapping.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Grid.h>
#include <dolfin/Settings.h>
#include <dolfin/bcfunction.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/Galerkin.h>
#include <dolfin/Equation.h>
#include <dolfin/EquationSystem.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Galerkin::Galerkin()
{
  // Will be created later (need to know the space dimension)
  element = 0;
  mapping = 0;
  quadrature = 0;

  // Using default method
  user = false;
}
//-----------------------------------------------------------------------------
Galerkin::Galerkin(FiniteElement& element,
						 Mapping&       mapping,
						 Quadrature&    quadrature)
{
  this->element    = &element;
  this->mapping    = &mapping;
  this->quadrature = &quadrature;

  // User specified method
  user = true;
}
//-----------------------------------------------------------------------------
Galerkin::~Galerkin()
{
  if ( !user ) {

	 if ( element )
		delete element;
	 element = 0;

	 if ( mapping )
		delete mapping;
	 mapping = 0;

	 if ( quadrature )
		delete quadrature;
	 quadrature = 0;
	 
  }
}
//-----------------------------------------------------------------------------
void Galerkin::assemble(Equation &equation, Grid &grid, Matrix &A, Vector &b)
{
  assembleLHS(equation, grid, A);
  assembleRHS(equation, grid, b);
  setBC(grid, A, b);
}
//-----------------------------------------------------------------------------
void Galerkin::assembleLHS(Equation &equation, Grid &grid, Matrix &A)
{  
  // Make sure that we have chosen trial and test spaces
  init(grid);
    
  // Allocate and reset matrix
  alloc(A, grid);

  // Write a message
  std::cout << "Assembling: system size is ";
  std::cout << A.size(0) << " x " << A.size(1) << "." << std::endl;

  // Iterate over all cells in the grid
  for (CellIterator cell(grid); !cell.end(); ++cell) {

	 // Update mapping
	 mapping->update(cell);

	 // Update element
	 element->update(mapping);
	 
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
  std::cout << "Assembling: " << A << std::endl;
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
	 
	 // Iterate over test functions
	 for (FiniteElement::TestFunctionIterator v(element); !v.end(); ++v)
		b(v.dof(cell)) += equation.rhs(v);
	 
  }
  
  // Write a message
  std::cout << "Assembling: " << b << std::endl;
}
//-----------------------------------------------------------------------------
void Galerkin::setBC(Grid &grid, Matrix &A, Vector &b)
{
  std::cout << "Setting boundary condition: Works only for nodal basis." << std::endl;
  
  BoundaryCondition bc;
  bcfunction bcf;
  Point p;
  
  // Get boundary condition function
  Settings::get("boundary condition", &bcf);

  // Write a message
  if ( !bcf )
	 std::cout << "Boundary conditions not specified." << std::endl;
  
  // Iterate over all nodes on the boundary
  for (NodeIterator node(grid); !node.end(); ++node) {

	 // Only set boundary condition for nodes on the boundary
	 if ( node->boundary() == -1 )
		continue;
	 
	 // Get coordinate
	 p = node->coord();
	 
	 // Get boundary condition
	 bc.update(p);
	 bcf(bc);

	 // Set boundary condition
	 switch ( bc.type() ) {
	 case BoundaryCondition::DIRICHLET:
		A.ident(node->id());
		b(node->id()) = bc.val();
		break;
	 case BoundaryCondition::NEUMANN:
		if ( bc.val() != 0.0 ) {
		  // FIXME: Use logging system
		  std::cout << "Error: Inhomogeneous Neumann boundary conditions not implemented." << std::endl;
		  exit(1);
		}
		break;
	 default:
		// FIXME: Use logging system
		std::cout << "Error: Unknown boundary condition." << std::endl;
		break;
	 }
	 
  }

}
//-----------------------------------------------------------------------------
void Galerkin::init(Grid &grid)
{
  // Check if the element has already been created
  if ( element )
	 return;

  // Create default finite element
  switch ( grid.type() ) {
  case Grid::TRIANGLES:
	 std::cout << "Using standard piecewise linears on triangles." << std::endl;
	 element    = new P1TriElement();
	 mapping    = new TriLinMapping();
	 quadrature = new TriangleMidpointQuadrature();
	 break;
  case Grid::TETRAHEDRONS:
	 std::cout << "Using standard piecewise linears on tetrahedrons." << std::endl;
	 element    = new P1TetElement();
	 mapping    = new TetLinMapping();
	 quadrature = new TetrahedronMidpointQuadrature();
	 break;
  default:
	 // FIXME: Use logging system
	 std::cout << "Error: No default spaces for this type of cells." << std::endl;
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
