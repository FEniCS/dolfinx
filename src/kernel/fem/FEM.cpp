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
  Quadrature& interior_quadrature(method.interiorQuadrature());
  Quadrature& boundary_quadrature(method.boundaryQuadrature());

  // Assemble matrix
  assemble(pde, mesh, A, element, map, interior_quadrature, boundary_quadrature);

  // Assemble vector
  assemble(pde, mesh, b, element, map, interior_quadrature, boundary_quadrature);
}
//-----------------------------------------------------------------------------
void FEM::assemble(PDE& pde, Mesh& mesh, Matrix& A)
{
  // Create default method
  FiniteElementMethod method(mesh.type(), pde.size());
  FiniteElement::Vector& element(method.element());
  Map& map(method.map());
  Quadrature& interior_quadrature(method.interiorQuadrature());
  Quadrature& boundary_quadrature(method.boundaryQuadrature());

  // Assemble matrix
  assemble(pde, mesh, A, element, map, interior_quadrature, boundary_quadrature);
}
//-----------------------------------------------------------------------------
void FEM::assemble(PDE& pde, Mesh& mesh, Vector& b)
{
  // Create default method
  FiniteElementMethod method(mesh.type(), pde.size());
  FiniteElement::Vector& element(method.element());
  Map& map(method.map());
  Quadrature& interior_quadrature(method.interiorQuadrature());
  Quadrature& boundary_quadrature(method.boundaryQuadrature());

  // Assemble vector
  assemble(pde, mesh, b, element, map, interior_quadrature, boundary_quadrature);
}
//-----------------------------------------------------------------------------
void FEM::assemble(PDE& pde, Mesh& mesh, Matrix& A,
		   FiniteElement::Vector& element, Map& map,
		   Quadrature& interior_quadrature, 
		   Quadrature& boundary_quadrature)
{
  // Allocate and reset matrix
  alloc(A, mesh, element);
  
  // Assemble interior
  assembleInterior(pde, mesh, A, element, map,
		   interior_quadrature, boundary_quadrature);
  
  // Assemble boundary
  assembleBoundary(pde, mesh, A, element, map,
		   interior_quadrature, boundary_quadrature);
  
  // Clear unused elements
  A.resize();

  // FIXME: This should be removed
  setBC(mesh, A, element);

  // Write a message
  cout << "Assembled: " << A << endl;
}
//-----------------------------------------------------------------------------
void FEM::assembleInterior(PDE& pde, Mesh& mesh, Matrix& A,
			   FiniteElement::Vector& element, Map& map,
			   Quadrature& interior_quadrature, 
			   Quadrature& boundary_quadrature)
{
  // Start a progress session
  Progress p("Assembling matrix (interior contribution)", mesh.noCells());
  
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update map
    map.update(*cell);
    
    // Update element
    for (unsigned int i = 0; i < element.size(); ++i)
      element(i)->update(map);
    
    // Update equation
    pde.updateLHS(element, map, interior_quadrature, boundary_quadrature);
    
    // Iterate over test and trial functions
    for (FiniteElement::Vector::TestFunctionIterator v(element); !v.end(); ++v)
      for (FiniteElement::Vector::TrialFunctionIterator u(element); !u.end(); ++u)
        A(v.dof(*cell), u.dof(*cell)) += pde.lhs(*u, *v);
    
    // Update progress
    p++;
  }
}
//-----------------------------------------------------------------------------
void FEM::assembleBoundary(PDE& pde, Mesh& mesh, Matrix& A,
			   FiniteElement::Vector& element, Map& map,
			   Quadrature& interior_quadrature, 
			   Quadrature& boundary_quadrature)
{
  // Remove for testing assembly on boundary
  return;

  // Check mesh type
  switch (mesh.type()) {
  case Mesh::triangles:
    assembleBoundaryTri(pde, mesh, A, element, map,
			interior_quadrature, boundary_quadrature);
    break;
  case Mesh::tetrahedrons:
    assembleBoundaryTet(pde, mesh, A, element, map,
			interior_quadrature, boundary_quadrature);
    break;
  default:
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void FEM::assembleBoundaryTri(PDE& pde, Mesh& mesh, Matrix& A,
			      FiniteElement::Vector& element, Map& map,
			      Quadrature& interior_quadrature, 
			      Quadrature& boundary_quadrature)
{
  // Copy code from assembleBoundaryTet when finished
}
//-----------------------------------------------------------------------------
void FEM::assembleBoundaryTet(PDE& pde, Mesh& mesh, Matrix& A,
			      FiniteElement::Vector& element, Map& map,
			      Quadrature& interior_quadrature, 
			      Quadrature& boundary_quadrature)
{
  cout << "Assembling over tetrahedral boundary." << endl;
  
  // Create boundary
  Boundary boundary(mesh);

  // Start a progress session
  Progress p("Assembling matrix (boundary contribution)", boundary.noFaces());  
    
  // Iterate over all faces in the boundary 
  for (FaceIterator face(boundary); !face.end(); ++face)
  {
    // Update map
    map.update(*face);
    
    // Get internal cell neighbor of face
    const Cell& cell = map.cell();

    // Update element
    for (unsigned int i = 0; i < element.size(); ++i)
      element(i)->update(map);
    
    // Update equation
    pde.updateLHS(element, map, interior_quadrature, boundary_quadrature);
    
    // Iterate over test and trial functions
    for (FiniteElement::Vector::TestFunctionIterator v(element); !v.end(); ++v)
      for (FiniteElement::Vector::TrialFunctionIterator u(element); !u.end(); ++u)
        A(v.dof(cell), u.dof(cell)) += pde.lhs(*u, *v);
    
    // Update progress
    p++;
  }
}
//-----------------------------------------------------------------------------
void FEM::assemble(PDE& pde, Mesh& mesh, Vector& b,
		   FiniteElement::Vector& element, Map& map,
		   Quadrature& interior_quadrature, 
		   Quadrature& boundary_quadrature)
{
  // Allocate and reset matrix
  alloc(b, mesh, element);
  
  // Assemble interior
  assembleInterior(pde, mesh, b, element, map,
		   interior_quadrature, boundary_quadrature);

  // Assemble boundary
  assembleBoundary(pde, mesh, b, element, map,
		   interior_quadrature, boundary_quadrature);

  // FIXME: This should be removed
  setBC(mesh, b, element);

  // Write a message
  cout << "Assembled: " << b << endl;
}
//-----------------------------------------------------------------------------
void FEM::assembleInterior(PDE& pde, Mesh& mesh, Vector& b,
			   FiniteElement::Vector& element, Map& map,
			   Quadrature& interior_quadrature, 
			   Quadrature& boundary_quadrature)
{
  // Start a progress session
  Progress p("Assembling vector (interior contribution)", mesh.noCells());  
  
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update map
    map.update(*cell);
    
    // Update equation
    pde.updateRHS(element, map, interior_quadrature, boundary_quadrature);
    
    // Iterate over test functions
    for (FiniteElement::Vector::TestFunctionIterator v(element); !v.end(); ++v)
      b(v.dof(*cell)) += pde.rhs(*v);
    
    // Update progress
    p++;
  }
}
//-----------------------------------------------------------------------------
void FEM::assembleBoundary(PDE& pde, Mesh& mesh, Vector& b,
			   FiniteElement::Vector& element, Map& map,
			   Quadrature& interior_quadrature, 
			   Quadrature& boundary_quadrature)
{
  // Check mesh type
  switch (mesh.type()) {
  case Mesh::triangles:
    assembleBoundaryTri(pde, mesh, b, element, map,
			interior_quadrature, boundary_quadrature);
    break;
  case Mesh::tetrahedrons:
    assembleBoundaryTet(pde, mesh, b, element, map,
			interior_quadrature, boundary_quadrature);
    break;
  default:
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void FEM::assembleBoundaryTri(PDE& pde, Mesh& mesh, Vector& b,
			      FiniteElement::Vector& element, Map& map,
			      Quadrature& interior_quadrature, 
			      Quadrature& boundary_quadrature)
{
  // Not implemented
}
//-----------------------------------------------------------------------------
void FEM::assembleBoundaryTet(PDE& pde, Mesh& mesh, Vector& b,
			      FiniteElement::Vector& element, Map& map,
			      Quadrature& interior_quadrature, 
			      Quadrature& boundary_quadrature)
{
  // Not implemented
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
      unsigned int i = v.dof(*cell);
      if ( i > imax )
	imax = i;
    }
    
    for (FiniteElement::TrialFunctionIterator u(element(0)); !u.end(); ++u)
    {
      unsigned int j = u.dof(*cell);
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
      unsigned int i = v.dof(*cell);
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
