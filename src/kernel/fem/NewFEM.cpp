// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <petsc/petscmat.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/NewPDE.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/Mesh.h>
#include <dolfin/NewMatrix.h>
#include <dolfin/NewVector.h>
#include <dolfin/Boundary.h>
#include <dolfin/NewFiniteElement.h>
#include <dolfin/NewFEM.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void NewFEM::assemble(BilinearForm& a, LinearForm& L,
		      NewMatrix& A, NewVector& b, Mesh& mesh,
		      const NewFiniteElement& element)
{
  // Start a progress session
  Progress p("Assembling matrix and vector (interior contributions)",
	     mesh.noCells());

  // Initialize element matrix/vector data block
  unsigned int n = element.spacedim();
  real* block_A = new real[n*n];
  real* block_b = new real[n];
  int* dofs = new int[n];

  // Initialize global matrix 
  // Max connectivity in NewMatrix::init() is assumed to 
  // be 50, alternatively use connectivity information to
  // minimize memory requirements.   
  unsigned int N = size(mesh, element);
  A.init(N, N);
  A = 0.0;
  b.init(N);
  b = 0.0;
  
  // Debug
  //   cout << "A inside:" << endl;
  //   A.disp();
  
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update form
    a.update(*cell, element);
    L.update(*cell, element);

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
void NewFEM::assemble(BilinearForm& a, NewMatrix& A, Mesh& mesh,
		      const NewFiniteElement& element)
{
  // Start a progress session
  Progress p("Assembling matrix (interior contribution)", mesh.noCells());

  // Initialize element matrix data block
  unsigned int n = element.spacedim();
  real* block = new real[n*n];
  int* dofs = new int[n];

  // Initialize global matrix 
  // Max connectivity in NewMatrix::init() is assumed to 
  // be 50, alternatively use connectivity information to
  // minimize memory requirements.   
  unsigned int N = size(mesh, element);
  A.init(N, N, 1);
  A = 0.0;

  // Debug
  //   A.apply();
  //   cout << "A inside:" << endl;
  //   A.disp();
  
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update form
    a.update(*cell, element);
    
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
void NewFEM::assemble(LinearForm& L, NewVector& b, Mesh& mesh,
		      const NewFiniteElement& element)
{
  // Start a progress session
  Progress p("Assembling vector (interior contribution)", mesh.noCells());

  // Initialize element vector data block
  unsigned int n = element.spacedim();
  real* block = new real[n];
  int* dofs = new int[n];

  // Initialize global vector 
  unsigned int N = size(mesh, element);
  b.init(N);
  b = 0.0;

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update form
    L.update(*cell, element);

    // Compute mapping from local to global degrees of freedom
    for (unsigned int i = 0; i < n; i++)
      dofs[i] = element.dof(i, *cell, mesh);
    
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
dolfin::uint NewFEM::size(Mesh& mesh, const NewFiniteElement& element)
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
