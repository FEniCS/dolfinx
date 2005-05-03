// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/Mesh.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/Boundary.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/FEM.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void FEM::assemble(BilinearForm& a, Matrix& A, Mesh& mesh)
{
  // Start a progress session
  Progress p("Assembling matrix (interior contribution)", mesh.noCells());

  // Get finite element
  // FIXME: Should not assume that test and trial elements are the same
  const FiniteElement& element = a.test();

  // Initialize element matrix data block
  unsigned int n = element.spacedim();
  real* block = new real[n*n];
  int* dofs = new int[n];

  // Initialize global matrix 
  // Max connectivity in Matrix::init() is assumed to 
  // be 50, alternatively use connectivity information to
  // minimize memory requirements.   
  unsigned int N = size(mesh, element);
  A.init(N, N, 1);
  A = 0.0;

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    //cout << "cell: " << (*cell).id() << endl;

    // Update form
    a.update(*cell);
    
    // Compute mapping from local to global degrees of freedom
    for (unsigned int i = 0; i < n; i++)
      dofs[i] = element.dof(i, *cell, mesh);

    // Compute element matrix
    a.interior(block);

    // Add element matrix to global matrix
    A.add(block, dofs, n, dofs, n);

    // Update progress
    p++;
  }
  
  // Complete assembly
  A.apply();

  // Delete data
  delete [] block;
  delete [] dofs;
}
//-----------------------------------------------------------------------------
void FEM::assemble(LinearForm& L, Vector& b, Mesh& mesh)
{
  // Start a progress session
  Progress p("Assembling vector (interior contribution)", mesh.noCells());

  // Get finite element
  const FiniteElement& test = L.test();

  // Initialize element vector data block
  unsigned int n = test.spacedim();
  real* block = new real[n];
  int* dofs = new int[n];

  // Initialize global vector 
  unsigned int N = size(mesh, test);
  b.init(N);
  b = 0.0;

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update form
    L.update(*cell);

    // Compute mapping from local to global degrees of freedom
    for (unsigned int i = 0; i < n; i++)
      dofs[i] = test.dof(i, *cell, mesh);
    
    // Compute element matrix
    L.interior(block);
    
    // Add element matrix to global matrix
    b.add(block, dofs, n);

    // Update progress
    p++;
  }
  
  // Complete assembly
  b.apply();

  // Delete data
  delete [] block;
  delete [] dofs;
}
//-----------------------------------------------------------------------------
void FEM::assemble(BilinearForm& a, LinearForm& L,
		   Matrix& A, Vector& b, Mesh& mesh)
{
  // Start a progress session
  Progress p("Assembling matrix and vector (interior contributions)",
	     mesh.noCells());

  // Get finite element
  // FIXME: Should not assume that test and trial element are the same
  const FiniteElement& element = a.test();

  // Initialize element matrix/vector data block
  unsigned int n = element.spacedim();
  real* block_A = new real[n*n];
  real* block_b = new real[n];
  int* dofs = new int[n];

  // Initialize global matrix 
  // Max connectivity in Matrix::init() is assumed to 
  // be 50, alternatively use connectivity information to
  // minimize memory requirements.   
  unsigned int N = size(mesh, element);
  A.init(N, N);
  A = 0.0;
  b.init(N);
  b = 0.0;
  
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update form
    a.update(*cell);
    L.update(*cell);

    // Compute mapping from local to global degrees of freedom
    for (unsigned int i = 0; i < n; i++)
      dofs[i] = element.dof(i, *cell, mesh);
   
    // Compute element matrix and vector 
    a.interior(block_A);
    L.interior(block_b);
    
    // Add element matrix to global matrix
    A.add(block_A, dofs, n, dofs, n);
    
    // Add element vector to global vector
    b.add(block_b, dofs, n);

    // Update progress
    p++;
  }
  
  // Complete assembly
  A.apply();
  b.apply();

  // Delete data
  delete [] block_A;
  delete [] block_b;
  delete [] dofs;
}
//-----------------------------------------------------------------------------
void FEM::assemble(BilinearForm& a, LinearForm& L,
		   Matrix& A, Vector& b, Mesh& mesh,
		   BoundaryCondition& bc)
{
  assemble(a, L, A, b, mesh);
  setBC(A, b, mesh, bc);
}
//-----------------------------------------------------------------------------
dolfin::uint FEM::size(Mesh& mesh, const FiniteElement& element)
{
  // Count the degrees of freedom (check maximum index)

  uint dofmax = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    for (uint i = 0; i < element.spacedim(); i++)
    {
      uint dof = element.dof(i, *cell, mesh);
      if ( dof > dofmax )
	dofmax = dof;
    }
  }

  return dofmax + 1;
}
//-----------------------------------------------------------------------------
void FEM::setBC(Matrix& A, Vector& b, Mesh& mesh, 
		BoundaryCondition& bc)
{
  // FIXME: This is a temporary implementation for linears. We need to iterate
  // FIXME: over all faces (edges) on the boundary and for each entity call
  // FIXME: the given boundary conditions for all dofs associated with the
  // FIXME: boundary. Only works for scalar linear elements.
  
  dolfin_info("Setting boundary conditions (works only for linears).");

  // Create boundary
  Boundary boundary(mesh);

  // Create boundary value
  BoundaryValue bv;

  // Allocate list of rows
  int* rows = new int[bc.numComponents() * boundary.noNodes()];

  // Iterate over all nodes on the boundary
  uint m = 0;
  for (NodeIterator node(boundary); !node.end(); ++node)
  {
    // Iterate over number of vector components
    for (uint c = 0; c < bc.numComponents(); c++)
    {
      // Get boundary condition
      if (bc.numComponents() > 1)
	bv = bc(node->coord(),c);
      else
	bv = bc(node->coord());
    
      // Set boundary condition if Dirichlet
      if ( bv.fixed )
      {
	uint dof = c * mesh.noNodes() + node->id();
	rows[m++] = dof;
	b(dof) = bv.value;
      }
    }
  }

  // Set rows of matrix to the identity matrix
  A.ident(rows, m);

  // Delete list of rows
  delete [] rows;
}
//-----------------------------------------------------------------------------
void FEM::lump(Matrix& M, Vector& m)
{
  m.init(M.size(0));

  Vector one(m);
  one = 1.0;

  M.mult(one, m);
}
//-----------------------------------------------------------------------------
