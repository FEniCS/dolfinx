// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Fredrik Bengzon and Johan Jansson, 2004.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/PDE.h>
#include <dolfin/Mesh.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/Map.h>
#include <dolfin/Quadrature.h>
#include <dolfin/FiniteElementMethod.h>
#include <dolfin/Boundary.h>
#include <dolfin/BCFunctionPointer.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/FEM.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void FEM::assemble(PDE& pde, Mesh& mesh, Matrix& A, Vector& b)
{
  // Create default method
  FiniteElementMethod method(mesh.type(), pde.size());
  FiniteElement::Vector& element(method.element());
  Map& map(method.map());
  Quadrature& quadrature(method.quadrature());

  // Assemble matrix
  assemble(pde, mesh, A, element, map, quadrature);

  // Assemble vector
  assemble(pde, mesh, b, element, map, quadrature);
}
//-----------------------------------------------------------------------------
void FEM::assemble(PDE& pde, Mesh& mesh, Matrix& A)
{
  // Create default method
  FiniteElementMethod method(mesh.type(), pde.size());
  FiniteElement::Vector& element(method.element());
  Map& map(method.map());
  Quadrature& quadrature(method.quadrature());

  // Assemble matrix
  assemble(pde, mesh, A, element, map, quadrature);
}
//-----------------------------------------------------------------------------
void FEM::assemble(PDE& pde, Mesh& mesh, Vector& b)
{
  // Create default method
  FiniteElementMethod method(mesh.type(), pde.size());
  FiniteElement::Vector& element(method.element());
  Map& map(method.map());
  Quadrature& quadrature(method.quadrature());

  // Assemble vector
  assemble(pde, mesh, b, element, map, quadrature);
}
//-----------------------------------------------------------------------------
void FEM::assemble(PDE& pde, Mesh& mesh, Matrix& A,
		   FiniteElement::Vector& element, Map& map,
		   Quadrature& quadrature)
{
  // Allocate and reset matrix
  alloc(A, mesh, element);
  
  // Start a progress session
  Progress p("Assembling left-hand side", mesh.noCells());  
  
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update map
    map.update(cell);
    
    // Update element
    for (unsigned int i = 0; i < element.size(); ++i)
      element(i)->update(map);
    
    // Update equation
    pde.updateLHS(element, cell, map, quadrature);
    
    // Iterate over test and trial functions
    for (FiniteElement::Vector::TestFunctionIterator v(element); !v.end(); ++v)
      for (FiniteElement::Vector::TrialFunctionIterator u(element); !u.end(); ++u)
        A(v.dof(cell), u.dof(cell)) += pde.lhs(*u, *v);
    
    // Update progress
    p++;
  }

  /*
  // Create boundary object 
  Boundary boundary(mesh);

  // Iterate over boundary 
  if (mesh.type() == tetrahedrons){

    // Start a progress session
    Progress p("Adding boundary conditions weakly", boundary.noFaces());  
    
    // Iterate over all faces in the boundary 
    for (FaceIterator face(boundary); !face.end(); ++face)
      {
	// Update map
	map.update(face.cell(0),face);
	
	// Update element
	for (unsigned int i = 0; i < element.size(); ++i)
	  element(i)->update(map);
	
	// Update equation
	pde.updateLHS(element, face.cell(0), map, quadrature);
	
	// Iterate over test and trial functions
	for (FiniteElement::Vector::TestFunctionIterator v(element); !v.end(); ++v)
	  for (FiniteElement::Vector::TrialFunctionIterator u(element); !u.end(); ++u)
	    A(v.dof(cell), u.dof(cell)) += pde.lhs(*u, *v);
	
	// Update progress
	p++;
      }

  } else if (mesh.type() == triangles){

    // Start a progress session
    Progress p("Adding boundary conditions weakly", boundary.noEdges());  
    
    // Iterate over all edges in the boundary
    for (EdgeIterator edge(mesh.boundary); !edge.end(); ++edge)
      {
	// Update map
	map.update(edge.cell(0),edge);
	
	// Update element
	for (unsigned int i = 0; i < element.size(); ++i)
	  element(i)->update(map);
	
	// Update equation
	pde.updateLHS(element, edge.cell(0), map, quadrature);
	
	// Iterate over test and trial functions
	for (FiniteElement::Vector::TestFunctionIterator v(element); !v.end(); ++v)
	  for (FiniteElement::Vector::TrialFunctionIterator u(element); !u.end(); ++u)
	    A(v.dof(cell), u.dof(cell)) += pde.lhs(*u, *v);
	
	// Update progress
	p++;
      }

  } else{

    dolfin_error("Cell type not implemented.");

  }    
  */

  // Clear unused elements
  A.resize();

  // FIXME: This should be removed
  setBC(mesh, A, element);

  // Write a message
  cout << "Assembled: " << A << endl;
}
//-----------------------------------------------------------------------------
void FEM::assemble(PDE& pde, Mesh& mesh, Vector& b,
		   FiniteElement::Vector& element, Map& map,
		   Quadrature& quadrature)
{
  // Allocate and reset matrix
  alloc(b, mesh, element);
  
  // Start a progress session
  Progress p("Assembling right-hand side", mesh.noCells());  
  
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update map
    map.update(cell);
    
    // Update equation
    pde.updateRHS(element, cell, map, quadrature);
    
    // Iterate over test functions
    for (FiniteElement::Vector::TestFunctionIterator v(element); !v.end(); ++v)
      b(v.dof(cell)) += pde.rhs(*v);
    
    // Update progress
    p++;
  }
  
  // FIXME: This should be removed
  setBC(mesh, b, element);

  // Write a message
  cout << "Assembled: " << b << endl;
}
//-----------------------------------------------------------------------------
void FEM::setBC(Mesh& mesh, Matrix& A, FiniteElement::Vector& element)
{
  cout << "Setting boundary condition: Works only for nodal basis." << endl;
  
  BoundaryCondition bc(element.size());
  bcfunction bcf;
  
  // Get boundary condition function
  bcf = dolfin_get("boundary condition");

  // Create boundary
  Boundary boundary(mesh);

  // Iterate over all nodes on the boundary
  for (NodeIterator node(boundary); !node.end(); ++node)
  {
    // Get boundary condition
    bc.update(node);
    bcf(bc);
    
    // Set boundary condition
    switch ( bc.type() ) {
    case BoundaryCondition::DIRICHLET:
      for (unsigned int i = 0; i < element.size(); i++)
      {
	A.ident(element.size() * node->id() + i);
      }

      //A.ident(node->id());
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
void FEM::setBC(Mesh& mesh, Vector& b, FiniteElement::Vector& element)
{
  cout << "Setting boundary condition: Works only for nodal basis." << endl;
  
  BoundaryCondition bc(element.size());
  bcfunction bcf;
  
  // Get boundary condition function
  bcf = dolfin_get("boundary condition");

  // Create boundary
  Boundary boundary(mesh);
  
  // Iterate over all nodes on the boundary
  for (NodeIterator node(boundary); !node.end(); ++node)
  {
    // Get boundary condition
    bc.update(node);
    bcf(bc);
    
    // Set boundary condition
    switch ( bc.type() ) {
    case BoundaryCondition::DIRICHLET:
      for (unsigned int i = 0; i < element.size(); i++)
	b(element.size() * node->id() + i) = bc.val(i);
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
void FEM::alloc(Matrix &A, Mesh &mesh, FiniteElement::Vector& element)
{
  // Count the degrees of freedom
  
  // For now we count the number of degrees of freedom of the first
  // element. Handling of heterogenous order of the elements is
  // deferred to future revisions.

  unsigned int imax = 0;
  unsigned int jmax = 0;
  
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    
    for (FiniteElement::TestFunctionIterator v(element(0)); !v.end(); ++v)
    {
      unsigned int i = v.dof(cell);
      if ( i > imax )
	imax = i;
    }
    
    for (FiniteElement::TrialFunctionIterator u(element(0)); !u.end(); ++u)
    {
      unsigned int j = u.dof(cell);
      if ( j > jmax )
	jmax = j;
    }

  }
  
  // Size of the matrix
  unsigned int m = (imax + 1) * element.size();
  unsigned int n = (jmax + 1) * element.size();
  
  // Initialise matrix
  if ( A.size(0) != m || A.size(1) != n )
    A.init(m, n);
}
//-----------------------------------------------------------------------------
void FEM::alloc(Vector &b, Mesh &mesh, FiniteElement::Vector& element)
{
  // Count the degrees of freedom
  unsigned int imax = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    for (FiniteElement::TestFunctionIterator v(element(0)); !v.end(); ++v)
    {
      unsigned int i = v.dof(cell);
      if ( i > imax )
	imax = i;
    }
  }
  
  // Size of vector
  unsigned int m = (imax + 1) * element.size();
  
  // Initialise vector
  if ( b.size() != m )
    b.init(m);
  
  // Set all entries to zero
  b = 0.0;
}
//-----------------------------------------------------------------------------
