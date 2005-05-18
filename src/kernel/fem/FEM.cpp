// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/AffineMap.h>
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

  // Get finite elements
  const FiniteElement& test_element = a.test();
  const FiniteElement& trial_element = a.trial();
  dolfin_assert(test_element.spacedim() == trial_element.spacedim());

  // Create affine map
  AffineMap map;

  // Initialize local data
  uint n = test_element.spacedim();
  real* block = new real[n*n];
  int* test_dofs = new int[n];
  int* trial_dofs = new int[n];

  // Initialize global matrix 
  uint N = size(mesh, test_element);
  A.init(N, N, 1);
  A = 0.0;

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update affine map
    map.update(*cell);

    // Update form
    a.update(map);
    
    // Compute maps from local to global degrees of freedom
    test_element.dofmap(test_dofs, *cell, mesh);
    trial_element.dofmap(trial_dofs, *cell, mesh);

    // Compute element matrix
    a.eval(block, map);

    // Add element matrix to global matrix
    A.add(block, test_dofs, n, trial_dofs, n);

    // Update progress
    p++;
  }
  
  // Complete assembly
  A.apply();

  // Delete data
  delete [] block;
  delete [] test_dofs;
  delete [] trial_dofs;
}
//-----------------------------------------------------------------------------
void FEM::assemble(LinearForm& L, Vector& b, Mesh& mesh)
{
  // Start a progress session
  Progress p("Assembling vector (interior contribution)", mesh.noCells());

  // Get finite element
  const FiniteElement& test_element = L.test();

  // Create affine map
  AffineMap map;

  // Initialize local data
  uint n = test_element.spacedim();
  real* block = new real[n];
  int* test_dofs = new int[n];

  // Initialize global vector 
  uint N = size(mesh, test_element);
  b.init(N);
  b = 0.0;

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update affine map
    map.update(*cell);

    // Update form
    L.update(map);

    // Compute map from local to global degrees of freedom
    test_element.dofmap(test_dofs, *cell, mesh);

    // Compute element matrix
    L.eval(block, map);
    
    // Add element matrix to global matrix
    b.add(block, test_dofs, n);

    // Update progress
    p++;
  }
  
  // Complete assembly
  b.apply();

  // Delete data
  delete [] block;
  delete [] test_dofs;
}
//-----------------------------------------------------------------------------
void FEM::assemble(BilinearForm& a, LinearForm& L,
		   Matrix& A, Vector& b, Mesh& mesh)
{
  // Start a progress session
  Progress p("Assembling matrix and vector (interior contributions)",
	     mesh.noCells());

  // Get finite elements
  const FiniteElement& test_element = a.test();
  const FiniteElement& trial_element = a.trial();
  dolfin_assert(test_element.spacedim() == trial_element.spacedim());

  // Create affine map
  AffineMap map;

  // Initialize element matrix/vector data block
  uint n = test_element.spacedim();
  real* block_A = new real[n*n];
  real* block_b = new real[n];
  int* test_dofs = new int[n];
  int* trial_dofs = new int[n];

  // Initialize global matrix 
  uint N = size(mesh, test_element);
  A.init(N, N);
  b.init(N);
  A = 0.0;
  b = 0.0;
  
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update affine map
    map.update(*cell);

    // Update forms
    a.update(map);
    L.update(map);

    // Compute maps from local to global degrees of freedom
    test_element.dofmap(test_dofs, *cell, mesh);
    trial_element.dofmap(trial_dofs, *cell, mesh);
   
    // Compute element matrix and vector 
    a.eval(block_A, map);
    L.eval(block_b, map);
    
    // Add element matrix to global matrix
    A.add(block_A, test_dofs, n, trial_dofs, n);
    
    // Add element vector to global vector
    b.add(block_b, test_dofs, n);

    // Update progress
    p++;
  }
  
  // Complete assembly
  A.apply();
  b.apply();

  // Delete data
  delete [] block_A;
  delete [] block_b;
  delete [] test_dofs;
  delete [] trial_dofs;
}
//-----------------------------------------------------------------------------
void FEM::assemble(BilinearForm& a, LinearForm& L,
		   Matrix& A, Vector& b, Mesh& mesh,
		   BoundaryCondition& bc)
{
  assemble(a, L, A, b, mesh);
  setBC(A, b, mesh, a.trial(), bc);
}
//-----------------------------------------------------------------------------
dolfin::uint FEM::size(Mesh& mesh, const FiniteElement& element)
{
  // Count the degrees of freedom (check maximum index)
  
  int* dofs = new int[element.spacedim()];
  int dofmax = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    element.dofmap(dofs, *cell, mesh);
    for (uint i = 0; i < element.spacedim(); i++)
    {
      if ( dofs[i] > dofmax )
	dofmax = dofs[i];
    }
  }
  delete [] dofs;

  return static_cast<uint>(dofmax + 1);
}
//-----------------------------------------------------------------------------
void FEM::setBC(Matrix& A, Vector& b, Mesh& mesh,
		const FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Applying Dirichlet boundary conditions.");

  // FIXME: Implement for tetrahedrons (iterate over faces) when working
  if ( mesh.type() == Mesh::tetrahedrons )
    dolfin_warning("Boundary conditions might not work for tetrahedrons.");

  // Create boundary
  Boundary boundary(mesh);

  // Create boundary value
  BoundaryValue bv;

  // Create affine map
  AffineMap map;

  // Allocate list of rows
  uint m = 0;
  int* rows = new int[b.size()];
  
  // Allocate local data
  uint n = element.spacedim();
  int* dofs = new int[n];
  uint* components = new uint[n];
  Point* points = new Point[n];

  // Iterate over all edges on the boundary
  for (EdgeIterator edge(boundary); !edge.end(); ++edge)
  {
    // Get cell containing the edge (pick first, should only be one)
    dolfin_assert(edge->noCellNeighbors() == 1);
    const Cell& cell = edge->cell(0);

    // Update affine map
    map.update(cell);

    // Compute map from local to global degrees of freedom
    element.dofmap(dofs, cell, mesh);

    // Compute map from local to global coordinates
    element.pointmap(points, components, map);

    // Set boundary conditions for dofs on the boundary
    for (uint i = 0; i < n; i++)
    {
      // Skip points that are not contained in edge
      const Point& point = points[i];
      if ( !(edge->contains(point)) )
	continue;

      // Get boundary condition
      if ( bc.numComponents() > 1 )
	bv = bc(point, components[i]);
      else
	bv = bc(point);
    
      // Set boundary condition if Dirichlet
      if ( bv.fixed )
      {
	int dof = dofs[i];
	rows[m++] = dof;
	b(dof) = bv.value;
      }
    }
  }

  // Set rows of matrix to the identity matrix
  A.ident(rows, m);

  // Delete data
  delete [] dofs;
  delete [] components;
  delete [] points;
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
