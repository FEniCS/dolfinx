// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_elements.h>
#include <dolfin/dolfin_quadrature.h>
#include <dolfin/utils.h>
#include <dolfin/Mapping.h>
#include <dolfin/TriLinMapping.h>
#include <dolfin/TetLinMapping.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Grid.h>
#include <dolfin/bcfunction.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/Galerkin.h>
#include <dolfin/PDE.h>
#include <dolfin/PDESystem.h>

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
void Galerkin::assemble(PDE& pde, Grid& grid, Matrix& A, Vector& b)
{
  assemble(pde, grid, A);
  assemble(pde, grid, b);
}
//-----------------------------------------------------------------------------
void Galerkin::assemble(PDE& pde, Grid& grid, Matrix& A)
{
  assembleLHS(pde, grid, A);
  setBC(grid, A);
}
//-----------------------------------------------------------------------------
void Galerkin::assemble(PDE& pde, Grid& grid, Vector& b)
{
  assembleRHS(pde, grid, b);
  setBC(grid, b);
}
//-----------------------------------------------------------------------------
void Galerkin::assembleLHS(PDE& pde, Grid& grid, Matrix& A)
{  
  // Make sure that we have chosen trial and test spaces
  init(grid);
  
  // Allocate and reset matrix
  alloc(A, grid);
  
  // Start a progress session
  Progress p("Assembling left-hand side", grid.noCells());  
  dolfin_info("Assembling: system size is %d x %d.", A.size(0), A.size(1));
  
  // Iterate over all cells in the grid
  for (CellIterator cell(grid); !cell.end(); ++cell) {
    
    // Update mapping
    mapping->update(cell);
    
    // Update element
    element->update(mapping);
    
    // Update equation
    pde.updateLHS(element, cell, mapping, quadrature);
    
    // Iterate over test and trial functions
    for (FiniteElement::TestFunctionIterator v(element); !v.end(); ++v)
      for (FiniteElement::TrialFunctionIterator u(element); !u.end(); ++u)
	A(v.dof(cell), u.dof(cell)) += pde.lhs(u, v);
    
    // Update progress
    p++;

  }
  
  // Clear unused elements
  A.resize();

  // Write a message
  cout << "Assembled: " << A << endl;
}
//-----------------------------------------------------------------------------
void Galerkin::assembleRHS(PDE& pde, Grid& grid, Vector& b)
{
  // Make sure that we have chosen trial and test spaces
  init(grid);
  
  // Allocate and reset matrix
  alloc(b, grid);
  
  // Start a progress session
  Progress p("Assembling right-hand side", grid.noCells());  
  
  // Iterate over all cells in the grid
  for (CellIterator cell(grid); !cell.end(); ++cell) {
    
    // Update mapping
    mapping->update(cell);
    
    // Update equation
    pde.updateRHS(element, cell, mapping, quadrature);
    
    // Iterate over test functions
    for (FiniteElement::TestFunctionIterator v(element); !v.end(); ++v)
      b(v.dof(cell)) += pde.rhs(v);
    
    // Update progress
    p++;

  }
  
  // Write a message
  cout << "Assembled: " << b << endl;
}
//-----------------------------------------------------------------------------
void Galerkin::setBC(Grid& grid, Matrix& A)
{
  cout << "Setting boundary condition: Works only for nodal basis." << endl;
  
  BoundaryCondition bc;
  bcfunction bcf;
  
  // Get boundary condition function
  bcf = dolfin_get("boundary condition");

  // Iterate over all nodes on the boundary
  for (NodeIterator node(grid); !node.end(); ++node) {
    
    // Only set boundary condition for nodes on the boundary
    if ( node->boundary() == -1 )
      continue;
    
    // Get boundary condition
    bc.update(node);
    bcf(bc);
    
    // Set boundary condition
    switch ( bc.type() ) {
    case BoundaryCondition::DIRICHLET:
      A.ident(node->id());
      break;
    case BoundaryCondition::NEUMANN:
      if ( bc.val() != 0.0 )
	dolfin_error("Inhomogeneous Neumann boundary conditions not implemented.");
      break;
    default:
      dolfin_error("Unknown boundary condition.");
      break;
    }
    
  }
  
}
//-----------------------------------------------------------------------------
void Galerkin::setBC(Grid& grid, Vector& b)
{
  cout << "Setting boundary condition: Works only for nodal basis." << endl;
  
  BoundaryCondition bc;
  bcfunction bcf;
  
  // Get boundary condition function
  bcf = dolfin_get("boundary condition");

  // Iterate over all nodes on the boundary
  for (NodeIterator node(grid); !node.end(); ++node) {
    
    // Only set boundary condition for nodes on the boundary
    if ( node->boundary() == -1 )
      continue;
    
    // Get boundary condition
    bc.update(node);
    bcf(bc);
    
    // Set boundary condition
    switch ( bc.type() ) {
    case BoundaryCondition::DIRICHLET:
      b(node->id()) = bc.val();
      break;
    case BoundaryCondition::NEUMANN:
      if ( bc.val() != 0.0 )
	dolfin_error("Inhomogeneous Neumann boundary conditions not implemented.");
      break;
    default:
      dolfin_error("Unknown boundary condition.");
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
    cout << "Using standard piecewise linears on triangles." << endl;
    element    = new P1TriElement();
    mapping    = new TriLinMapping();
    quadrature = new TriangleMidpointQuadrature();
    break;
  case Grid::TETRAHEDRONS:
    cout << "Using standard piecewise linears on tetrahedrons." << endl;
    element    = new P1TetElement();
    mapping    = new TetLinMapping();
    quadrature = new TetrahedronMidpointQuadrature();
    break;
  default:
    dolfin_error("No default spaces for this type of cell.");
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
