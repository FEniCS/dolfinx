// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_element.h>
#include <dolfin/dolfin_quadrature.h>
#include <dolfin/utils.h>
#include <dolfin/Map.h>
#include <dolfin/P1TriMap.h>
#include <dolfin/P1TetMap.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Mesh.h>
#include <dolfin/Boundary.h>
#include <dolfin/BCFunctionPointer.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/Galerkin.h>
#include <dolfin/PDE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Galerkin::Galerkin()
{
  // Will be created later (need to know the space dimension)
  element = 0;
  map = 0;
  quadrature = 0;

  // Using default method
  user = false;
}
//-----------------------------------------------------------------------------
Galerkin::Galerkin(FiniteElement& element,
		   Map&           map,
		   Quadrature&    quadrature)
{
  this->element = &element;
  this->map = &map;
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
    
    if ( map )
      delete map;
    map = 0;
    
    if ( quadrature )
      delete quadrature;
    quadrature = 0;
    
  }
}
//-----------------------------------------------------------------------------
void Galerkin::assemble(PDE& pde, Mesh& mesh, Matrix& A, Vector& b)
{
  assemble(pde, mesh, A);
  assemble(pde, mesh, b);
}
//-----------------------------------------------------------------------------
void Galerkin::assemble(PDE& pde, Mesh& mesh, Matrix& A)
{
  assembleLHS(pde, mesh, A);
  setBC(mesh, A);
}
//-----------------------------------------------------------------------------
void Galerkin::assemble(PDE& pde, Mesh& mesh, Vector& b)
{
  assembleRHS(pde, mesh, b);
  setBC(mesh, b);
}
//-----------------------------------------------------------------------------
void Galerkin::assembleLHS(PDE& pde, Mesh& mesh, Matrix& A)
{  
  // Make sure that we have chosen trial and test spaces
  init(mesh);
  
  // Allocate and reset matrix
  alloc(A, mesh);
  
  // Start a progress session
  Progress p("Assembling left-hand side", mesh.noCells());  
  dolfin_info("Assembling: system size is %d x %d.", A.size(0), A.size(1));
  
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell) {
    
    // Update map
    map->update(cell);
    
    // Update element
    element->update(map);
    
    // Update equation
    pde.updateLHS(element, cell, map, quadrature);
    
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
void Galerkin::assembleRHS(PDE& pde, Mesh& mesh, Vector& b)
{
  // Make sure that we have chosen trial and test spaces
  init(mesh);
  
  // Allocate and reset matrix
  alloc(b, mesh);
  
  // Start a progress session
  Progress p("Assembling right-hand side", mesh.noCells());  
  
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell) {
    
    // Update map
    map->update(cell);
    
    // Update equation
    pde.updateRHS(element, cell, map, quadrature);
    
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
void Galerkin::setBC(Mesh& mesh, Matrix& A)
{
  cout << "Setting boundary condition: Works only for nodal basis." << endl;
  
  BoundaryCondition bc;
  bcfunction bcf;
  
  // Get boundary condition function
  bcf = dolfin_get("boundary condition");

  // Create boundary
  Boundary boundary(mesh);

  // Iterate over all nodes on the boundary
  for (NodeIterator node(boundary); !node.end(); ++node) {

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
void Galerkin::setBC(Mesh& mesh, Vector& b)
{
  cout << "Setting boundary condition: Works only for nodal basis." << endl;
  
  BoundaryCondition bc;
  bcfunction bcf;
  
  // Get boundary condition function
  bcf = dolfin_get("boundary condition");

  // Create boundary
  Boundary boundary(mesh);
  
  // Iterate over all nodes on the boundary
  for (NodeIterator node(boundary); !node.end(); ++node) {
   
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
void Galerkin::init(Mesh &mesh)
{
  // Check if the element has already been created
  if ( element )
	 return;

  // Create default finite element
  switch ( mesh.type() ) {
  case Mesh::triangles:
    cout << "Using standard piecewise linears on triangles." << endl;
    element = new P1TriElement();
    map = new P1TriMap();
    quadrature = new TriangleMidpointQuadrature();
    break;
  case Mesh::tetrahedrons:
    cout << "Using standard piecewise linears on tetrahedrons." << endl;
    element = new P1TetElement();
    map = new P1TetMap();
    quadrature = new TetrahedronMidpointQuadrature();
    break;
  default:
    dolfin_error("No default spaces for this type of cell.");
  }
}
//-----------------------------------------------------------------------------
void Galerkin::alloc(Matrix &A, Mesh &mesh)
{
  // Count the degrees of freedom
  
  unsigned int imax = 0;
  unsigned int jmax = 0;

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    
    for (FiniteElement::TestFunctionIterator v(element); !v.end(); ++v)
    {
      unsigned int i = v.dof(cell);
      if ( i > imax )
	imax = i;
    }

    for (FiniteElement::TrialFunctionIterator u(element); !u.end(); ++u)
    {
      unsigned int j = u.dof(cell);
      if ( j > jmax )
	jmax = j;
    }

  }
  
  // Size of the matrix
  unsigned int m = imax + 1;
  unsigned int n = jmax + 1;
  
  // Initialise matrix
  if ( A.size(0) != m || A.size(1) != n )
    A.init(m, n);
}
//-----------------------------------------------------------------------------
void Galerkin::alloc(Vector &b, Mesh &mesh)
{
  // Count the degrees of freedom

  unsigned int imax = 0;
  
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    for (FiniteElement::TestFunctionIterator v(element); !v.end(); ++v)
    {
      unsigned int i = v.dof(cell);
      if ( i > imax )
	imax = i;
    }
  }
  
  // Size of vector
  unsigned int m = imax + 1;
  
  // Initialise vector
  if ( b.size() != m )
    b.init(m);
  
  // Set all entries to zero
  b = 0.0;
}
//-----------------------------------------------------------------------------
