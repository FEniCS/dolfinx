// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/NewPDE.h>
#include <dolfin/Mesh.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/NewFEM.h>
#include <dolfin/Boundary.h>
#include <dolfin/BCFunctionPointer.h>
#include <dolfin/BoundaryCondition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void NewFEM::assemble(NewPDE& pde, Mesh& mesh, Matrix& A, Vector& b)
{
  // Assemble matrix
  assemble(pde, mesh, A);

  // Assemble vector
  assemble(pde, mesh, b);
}
//-----------------------------------------------------------------------------
void NewFEM::assemble(NewPDE& pde, Mesh& mesh, Matrix& A)
{
  // Allocate and reset matrix
  //  alloc(A, mesh, pde);
  
  // Assemble interior
  assembleInterior(pde, mesh, A);
  
  // Assemble boundary
  assembleBoundary(pde, mesh, A);
  
  // Clear unused elements
  A.resize();

  // FIXME: This should be removed
  //setBC(mesh, A, pde);

  // Write a message
  cout << "Assembled: " << A << endl; 
}
//-----------------------------------------------------------------------------
void NewFEM::assemble(NewPDE& pde, Mesh& mesh, Vector& b)
{
  // Allocate and reset matrix
  //alloc(b, mesh, pde);
  
  // Assemble interior
  assembleInterior(pde, mesh, b);

  // Assemble boundary
  assembleBoundary(pde, mesh, b);

  // FIXME: This should be removed
  //setBC(mesh, b, pde);

  // Write a message
  cout << "Assembled: " << b << endl;
}
//-----------------------------------------------------------------------------
void NewFEM::assembleInterior(NewPDE& pde, Mesh& mesh, Matrix& A)
{
  // Start a progress session
  Progress p("Assembling matrix (interior contribution)", mesh.noCells());
  
  //NewArray< NewArray<real> > AK(pde.size(),pde.size());
  NewArray< NewArray<real> > AK;

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update PDE
    pde.update(*cell);
    
    // Compute element matrix    
    pde.interiorElementMatrix(AK);

    // Insert element matrix into global matrix
    for (unsigned int i = 0; i < pde.size(); i++) 
      for (unsigned int j = 0; j < pde.size(); j++) 
	A(pde.dof(i,*cell),pde.dof(j,*cell)) += AK[i][j];
    
    // Update progress
    p++;
  }
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBoundary(NewPDE& pde, Mesh& mesh, Matrix& A)
{
  // Check mesh type
  switch (mesh.type()) {
  case Mesh::triangles:
    assembleBoundaryTri(pde, mesh, A);
    break;
  case Mesh::tetrahedrons:
    assembleBoundaryTet(pde, mesh, A);
    break;
  default:
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBoundaryTri(NewPDE& pde, Mesh& mesh, Matrix& A)
{
  // Create boundary
  Boundary boundary(mesh);

  // Start a progress session
  Progress p("Assembling matrix (boundary contribution)", boundary.noEdges());
    
  // Iterate over all edges in the boundary 
  for (EdgeIterator edge(boundary); !edge.end(); ++edge)
  {
    /*
    // Update PDE
    pde.update(*cell);
    
    // Compute element matrix    
    pde.interiorElementMatrix(AK);

    // Insert element matrix into global matrix
    for (unsigned int i = 0; i < pde.size(); i++) 
      for (unsigned int j = 0; j < pde.size(); j++) 
	A(pde.dof(i,*cell),pde.dof(j,*cell)) += AK[i][j];
    */

    // Update progress
    p++;
  }
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBoundaryTet(NewPDE& pde, Mesh& mesh, Matrix& A)
{
  // Create boundary
  Boundary boundary(mesh);

  // Start a progress session
  Progress p("Assembling matrix (boundary contribution)", boundary.noFaces());
    
  // Iterate over all faces in the boundary 
  for (FaceIterator face(boundary); !face.end(); ++face)
  {
    /*
    // Update PDE
    pde.update(*cell);
    
    // Compute element matrix    
    pde.interiorElementMatrix(AK);

    // Insert element matrix into global matrix
    for (unsigned int i = 0; i < pde.size(); i++) 
      for (unsigned int j = 0; j < pde.size(); j++) 
	A(pde.dof(i,*cell),pde.dof(j,*cell)) += AK[i][j];
    */

    // Update progress
    p++;
  }
}
//-----------------------------------------------------------------------------
void NewFEM::assembleInterior(NewPDE& pde, Mesh& mesh, Vector& b)
{
  // Start a progress session
  Progress p("Assembling vector (interior contribution)", mesh.noCells());  
  
  NewArray<real> bK(pde.size());

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update PDE
    pde.update(*cell);
    
    // Compute element matrix    
    pde.interiorElementVector(bK);

    // Insert element matrix into global matrix
    for (unsigned int i = 0; i < pde.size(); i++) 
      b(pde.dof(i,*cell)) += bK[i];
    
    // Update progress
    p++;
  }
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBoundary(NewPDE& pde, Mesh& mesh, Vector& b)
{
  // Check mesh type
  switch (mesh.type()) {
  case Mesh::triangles:
    assembleBoundaryTri(pde, mesh, b);
    break;
  case Mesh::tetrahedrons:
    assembleBoundaryTet(pde, mesh, b);
    break;
  default:
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBoundaryTri(NewPDE& pde, Mesh& mesh, Vector& b)
{
  // Create boundary
  Boundary boundary(mesh);
  
  // Start a progress session
  Progress p("Assembling matrix (boundary contribution)", boundary.noEdges());
    
  // Iterate over all edges in the boundary 
  for (EdgeIterator edge(boundary); !edge.end(); ++edge)
  {
    /*
    // Update PDE
    pde.update(*cell);
    
    // Compute element matrix    
    pde.interiorElementVector(bK);

    // Insert element matrix into global matrix
    for (unsigned int i = 0; i < pde.size(); i++) 
      b(pde.dof(i,*cell)) += bK[i];
    */

    // Update progress
    p++;
  }
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBoundaryTet(NewPDE& pde, Mesh& mesh, Vector& b)
{
  // Create boundary
  Boundary boundary(mesh);
  
  // Start a progress session
  Progress p("Assembling matrix (boundary contribution)", boundary.noFaces());  
    
  // Iterate over all faces in the boundary 
  for (FaceIterator face(boundary); !face.end(); ++face)
  {
    /*
    // Update PDE
    pde.update(*cell);
    
    // Compute element matrix    
    pde.interiorElementVector(bK);

    // Insert element matrix into global matrix
    for (unsigned int i = 0; i < pde.size(); i++) 
      b(pde.dof(i,*cell)) += bK[i];
    */

    // Update progress
    p++;
  }
}
//-----------------------------------------------------------------------------
void NewFEM::setBC(Mesh& mesh, Matrix& A, NewPDE& pde)
{
  cout << "Setting boundary condition: Works only for nodal basis." << endl;
  
  BoundaryCondition bc(pde.dim());
  bcfunction bcf;
  
  // Get boundary condition function
  bcf = dolfin_get("boundary condition");

  // Create boundary
  Boundary boundary(mesh);

  // Iterate over all nodes on the boundary
  for (NodeIterator node(boundary); !node.end(); ++node)
  {
    // Get boundary condition
    //bc.update(node);
    bcf(bc);
    
    // Set boundary condition
    switch ( bc.type() ) {
    case BoundaryCondition::DIRICHLET:
      for (unsigned int i = 0; i < pde.dim(); i++)
      {
	A.ident(pde.dim() * node->id() + i);
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
void NewFEM::setBC(Mesh& mesh, Vector& b, NewPDE& pde)
{
  cout << "Setting boundary condition: Works only for nodal basis." << endl;
  
  BoundaryCondition bc(pde.dim());
  bcfunction bcf;
  
  // Get boundary condition function
  bcf = dolfin_get("boundary condition");

  // Create boundary
  Boundary boundary(mesh);
  
  // Iterate over all nodes on the boundary
  for (NodeIterator node(boundary); !node.end(); ++node)
  {
    // Get boundary condition
    //bc.update(node);
    bcf(bc);
    
    // Set boundary condition
    switch ( bc.type() ) {
    case BoundaryCondition::DIRICHLET:
      for (unsigned int i = 0; i < pde.dim(); i++)
	b(pde.dim() * node->id() + i) = bc.val(i);
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
void NewFEM::alloc(Matrix &A, Mesh &mesh, NewPDE& pde)
{
  // Count the degrees of freedoml
  // Assume square matrix. 

  unsigned int imax = 0;
  //unsigned int jmax = 0;
  
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    
    for (unsigned int sf = 0; sf < pde.size(); sf++)
    {
      unsigned int i = pde.dof(sf,*cell);
      if ( i > imax )
	imax = i;
    }
    
  }
  
  // Size of the matrix
  unsigned int m = (imax + 1) * pde.dim();
  //unsigned int n = (jmax + 1) * pde.dim();
  
  // Initialise matrix
  if ( A.size(0) != m || A.size(1) != m )
    A.init(m, m);
  //if ( A.size(0) != m || A.size(1) != n )
  //  A.init(m, n);

  // Set all entries to zero (for repeated assembling)
  A = 0.0;
}
//-----------------------------------------------------------------------------
void NewFEM::alloc(Vector &b, Mesh &mesh, NewPDE& pde)
{
  // Count the degrees of freedom
  unsigned int imax = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    for (unsigned int sf = 0; sf < pde.size(); sf++)
    {
      unsigned int i = pde.dof(sf,*cell);
      if ( i > imax )
	imax = i;
    }
    
  }
  
  // Size of vector
  unsigned int m = (imax + 1) * pde.dim();
  
  // Initialise vector
  if ( b.size() != m )
    b.init(m);
  
  // Set all entries to zero
  b = 0.0;
}
//-----------------------------------------------------------------------------
