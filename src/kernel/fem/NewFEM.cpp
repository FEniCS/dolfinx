// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/NewPDE.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/Mesh.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/Boundary.h>
#include <dolfin/NewFiniteElement.h>
#include <dolfin/NewFEM.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void NewFEM::assemble(NewPDE& pde, Mesh& mesh, Matrix& A, Vector& b)
{
  // Assemble matrix
  assemble(pde.a, mesh, A);

  // Assemble vector
  assemble(pde.L, mesh, b);
}
//-----------------------------------------------------------------------------
void NewFEM::assemble(BilinearForm& a, Mesh& mesh, Matrix& A)
{
  cout << "size = " << a.element.spacedim() << endl;


  // Allocate and reset matrix
  alloc(a.element, mesh, A);
  
  // Assemble interior contribution
  assembleInterior(a, mesh, A);
  
  // Assemble boundary contribution
  assembleBoundary(a, mesh, A);
  
  // Clear unused elements
  A.resize();

  // Write a message
  cout << "Assembled: " << A << endl; 
}
//-----------------------------------------------------------------------------
void NewFEM::assemble(LinearForm& L, Mesh& mesh, Vector& b)
{
  // Allocate and reset vector
  alloc(L.element, mesh, b);
  
  // Assemble interior contribution
  assembleInterior(L, mesh, b);

  // Assemble boundary contribution
  assembleBoundary(L, mesh, b);

  // Write a message
  cout << "Assembled: " << b << endl;
}
//-----------------------------------------------------------------------------
void NewFEM::assembleInterior(BilinearForm& a, Mesh& mesh, Matrix& A)
{
  // Start a progress session
  Progress p("Assembling matrix (interior contribution)", mesh.noCells());
  
  // Initialize finite element and element matrix
  const NewFiniteElement& element = a.element;
  real** AK = allocElementMatrix(element);

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update form
    a.update(*cell);
    
    // Compute element matrix
    bool result = a.interior(AK);

    // Check if we should skip the remaining cells
    if ( !result )
    {
      dolfin_info("Form does not contain a contribution from interior of domain.");
      dolfin_info("Skipping remaining cells.");
      p = 1.0;
      break;
    }

    // Add nonzero entries to global matrix
    for (unsigned int n = 0; n < a.nonzero.size(); n++)
    {
      const IndexPair& index = a.nonzero[n];
      A(element.dof(index.i, *cell), element.dof(index.j, *cell)) += AK[index.i][index.j];
    }
    
    // Update progress
    p++;
  }
  
  // Delete element matrix
  freeElementMatrix(AK, element);
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBoundary(BilinearForm& a, Mesh& mesh, Matrix& A)
{
  // Check mesh type
  switch (mesh.type()) {
  case Mesh::triangles:
    assembleBoundaryTri(a, mesh, A);
    break;
  case Mesh::tetrahedrons:
    assembleBoundaryTet(a, mesh, A);
    break;
  default:
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBoundaryTri(BilinearForm& a, Mesh& mesh, Matrix& A)
{
  // FIXME: Not implemented
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBoundaryTet(BilinearForm& a, Mesh& mesh, Matrix& A)
{
  // FIXME: Not implemented
}
//-----------------------------------------------------------------------------
void NewFEM::assembleInterior(LinearForm& L, Mesh& mesh, Vector& b)
{
  // Start a progress session
  Progress p("Assembling matrix (interior contribution)", mesh.noCells());
  
  // Initialize finite element and element vector
  const NewFiniteElement& element = L.element;
  real* bK = allocElementVector(element);

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update form
    L.update(*cell);
    
    // Compute element vector
    bool result = L.interior(bK);

    // Check if we should skip the remaining cells
    if ( !result )
    {
      dolfin_info("Form does not contain a contribution from interior of domain.");
      dolfin_info("Skipping remaining cells.");
      p = 1.0;
      break;
    }

    // Add entries to global vector
    for (unsigned int i = 0; i < element.spacedim(); i++)
      b(element.dof(i, *cell)) += bK[i];
  }
  
  // Delete element vector
  freeElementVector(bK, element);
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBoundary(LinearForm& L, Mesh& mesh, Vector& b)
{
  // Check mesh type
  switch (mesh.type()) {
  case Mesh::triangles:
    assembleBoundaryTri(L, mesh, b);
    break;
  case Mesh::tetrahedrons:
    assembleBoundaryTet(L, mesh, b);
    break;
  default:
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBoundaryTri(LinearForm& L, Mesh& mesh, Vector& b)
{
  // FIXME: Not implemented
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBoundaryTet(LinearForm& L, Mesh& mesh, Vector& b)
{
  // FIXME: Not implemented
}
//-----------------------------------------------------------------------------
void NewFEM::alloc(const NewFiniteElement& element, Mesh& mesh, Matrix& A)
{
  // Count the degrees of freedom (check maximum index)
  unsigned int dofmax = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    for (unsigned int i = 0; i < element.spacedim(); i++)
    {
      unsigned int dof = element.dof(i, *cell);
      if ( dof > dofmax )
	dofmax = dof;
    }
  }
  
  // Initialise matrix
  dofmax++;
  if ( A.size(0) != dofmax || A.size(1) != dofmax )
    A.init(dofmax, dofmax);

  // Set all entries to zero (for repeated assembly)
  A = 0.0;
}
//-----------------------------------------------------------------------------
void NewFEM::alloc(const NewFiniteElement& element, Mesh& mesh, Vector& b)
{
  // Count the degrees of freedom (check maximum index)
  unsigned int dofmax = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    for (unsigned int i = 0; i < element.spacedim(); i++)
    {
      unsigned int dof = element.dof(i, *cell);
      if ( dof > dofmax )
	dofmax = dof;
    }
  }
  
  // Initialise vector
  dofmax++;
  if ( b.size() != dofmax )
    b.init(dofmax);

  // Set all entries to zero (for repeated assembly)
  b = 0.0;
}
//-----------------------------------------------------------------------------
real** NewFEM::allocElementMatrix(const NewFiniteElement& element)
{
  cout << "Allocating local matrix of size: " << element.spacedim() << endl;


  // Allocate element matrix
  real** AK = new real*[element.spacedim()];
  for (unsigned int i = 0; i < element.spacedim(); i++)
    AK[i] = new real [element.spacedim()];
  
  // Set all entries to zero                                                    
  for (unsigned int i = 0; i < element.spacedim(); i++)
    for (unsigned int j = 0; j < element.spacedim(); j++)
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
