// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <petsc/petscmat.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/NewPDE.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/Mesh.h>
#include <dolfin/Matrix.h>
#include <dolfin/NewMatrix.h>
#include <dolfin/NewVector.h>
#include <dolfin/Vector.h>
#include <dolfin/Boundary.h>
#include <dolfin/NewFiniteElement.h>
#include <dolfin/NewFEM.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
real** NewFEM::allocElementMatrix(const NewFiniteElement& element)
{
  unsigned int ncomponents = 0;

  if(element.rank() == 0)
  {
    ncomponents = 1;
  }
  else if(element.rank() == 1)
  {
    ncomponents = element.tensordim(0);
  }

  // Allocate element matrix
  real** AK = new real*[element.spacedim() * ncomponents];
  for (unsigned int i = 0; i < element.spacedim() * ncomponents; i++)
    AK[i] = new real [element.spacedim() * ncomponents];
  
  // Set all entries to zero                                                    
  for (unsigned int i = 0; i < element.spacedim() * ncomponents; i++)
    for (unsigned int j = 0; j < element.spacedim() * ncomponents; j++)
      AK[i][j] = 0.0;

  return AK;
}
//-----------------------------------------------------------------------------
real* NewFEM::allocElementVector(const NewFiniteElement& element)
{
  // Allocate element vector
  real* bK = new real[element.spacedim()];

  // Set all entries to zero
  for (unsigned int i = 0; i < element.spacedim(); i++)
    bK[i] = 0.0;

  return bK;
}
//-----------------------------------------------------------------------------
void NewFEM::freeElementMatrix(real**& AK, const NewFiniteElement& element)
{
  for (unsigned int i = 0; i < element.spacedim(); i++)
    delete [] AK[i];
  delete [] AK;
  AK = 0;
}
//-----------------------------------------------------------------------------
void NewFEM::freeElementVector(real*& bK, const NewFiniteElement& element)
{
  delete [] bK;
  bK = 0;
}
//-----------------------------------------------------------------------------
void NewFEM::assemble(BilinearForm& a, LinearForm& L, Mesh& mesh, 
		      NewMatrix& A, NewVector& b) 
{

  // Start a progress session
  Progress p("Assembling matrix and vector (interior contributions)", mesh.noCells());

  // Initialize finite element and element matrix and vector
  const NewFiniteElement& element = a.element;
  real** AK = allocElementMatrix(element);
  real*  bK = allocElementVector(element);

  unsigned int n = element.spacedim();
  real* block = new real[n*n];
  int* dofs = new int[n];

  // Initialize global matrix 
  // Max connectivity in NewMatrix::init() is assumed to 
  // be 50, alternatively use connectivity information to
  // minimize memory requirements. 
  unsigned int N = size(mesh, element);
  A.init(N, N, element.spacedim());
  A = 0.0;
  b.init(N);
  b = 0.0;

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update form
    a.update(*cell);
    L.update(*cell);
   
    // Compute element matrix and vector 
    a.interior(AK);
    L.interior(bK);

    // Copy values to one array
    unsigned int pos = 0;
    for (unsigned int i = 0; i < n; i++)
      for (unsigned int j = 0; j < n; j++)
	block[pos++] = AK[i][j];

    // Copy values to one array
    pos = 0;
    for (unsigned int i = 0; i < n; i++)
      block[pos++] = bK[i];

    // Compute mapping from local to global degrees of freedom
    for (unsigned int i = 0; i < n; i++)
      dofs[i] = element.dof(i, *cell);
    
    // Add element matrix to global matrix
    A.add(block, dofs, n, dofs, n);
    
    // Add element matrix to global matrix
    b.add(block, dofs, n);

    // Update progress
    p++;
  }
  
  // Complete assembly
  A.apply();
  b.apply();

  delete [] block;
  delete [] dofs;

  // Delete element matrix
  freeElementMatrix(AK, element);
  freeElementVector(bK, element);

}
//-----------------------------------------------------------------------------
void NewFEM::assemble(BilinearForm& a, Mesh& mesh, NewMatrix& A)
{

  // Start a progress session
  Progress p("Assembling matrix (interior contribution)", mesh.noCells());

  // Initialize finite element and element matrix
  const NewFiniteElement& element = a.element;
  real** AK = allocElementMatrix(element);

  unsigned int n = element.spacedim();
  real* block = new real[n*n];
  int* dofs = new int[n];

  // Initialize global matrix 
  // Max connectivity in NewMatrix::init() is assumed to 
  // be 50, alternatively use connectivity information to
  // minimize memory requirements. 
  unsigned int N = size(mesh, element);
  A.init(N, N, element.spacedim());
  A = 0.0;

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update form
    a.update(*cell);
    
    // Compute element matrix
    a.interior(AK);

    // Copy values to one array
    unsigned int pos = 0;
    for (unsigned int i = 0; i < n; i++)
      for (unsigned int j = 0; j < n; j++)
	block[pos++] = AK[i][j];

    // Compute mapping from local to global degrees of freedom
    for (unsigned int i = 0; i < n; i++)
      dofs[i] = element.dof(i, *cell);
    
    // Add element matrix to global matrix
    A.add(block, dofs, n, dofs, n);

    // Update progress
    p++;
  }
  
  // Complete assembly
  A.apply();

  delete [] block;
  delete [] dofs;

  // Delete element matrix
  freeElementMatrix(AK, element);

}
//-----------------------------------------------------------------------------
void NewFEM::assemble(LinearForm& L, Mesh& mesh, NewVector& b)
{
  // Start a progress session
  Progress p("Assembling vector (interior contribution)", mesh.noCells());

  // Initialize finite element and element matrix
  const NewFiniteElement& element = L.element;
  real* bK = allocElementVector(element);

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
    L.update(*cell);
    
    // Compute element matrix
    L.interior(bK);

    // Copy values to one array
    unsigned int pos = 0;
    for (unsigned int i = 0; i < n; i++)
      block[pos++] = bK[i];

    // Compute mapping from local to global degrees of freedom
    for (unsigned int i = 0; i < n; i++)
      dofs[i] = element.dof(i, *cell);
    
    // Add element matrix to global matrix
    b.add(block, dofs, n);

    // Update progress
    p++;
  }
  
  // Complete assembly
  b.apply();

  delete [] block;
  delete [] dofs;

  // Delete element matrix
  freeElementVector(bK, element);
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
      uint dof = element.dof(i, *cell);
      if ( dof > dofmax )
	dofmax = dof;
    }
  }
  return dofmax + 1;
}
//-----------------------------------------------------------------------------

