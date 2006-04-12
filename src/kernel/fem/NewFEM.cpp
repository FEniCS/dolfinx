// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Andy Terrel 2005.
// Modified by Garth N. Wells 2005, 2006.
//
// First added:  2004-05-19
// Last changed: 2006-04-12

#include <dolfin/dolfin_log.h>
#include <dolfin/timing.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/Boundary.h>
#include <dolfin/NewFEM.h>


using namespace dolfin;

//-----------------------------------------------------------------------------
void NewFEM::assemble(BilinearForm& a, Matrix& A, Mesh& mesh)
{
  LinearForm* L = 0;
  Vector* b = 0;
  assemble_test(&a, L, A, *b, mesh);
}
//-----------------------------------------------------------------------------
void NewFEM::assemble(LinearForm& L, Vector& b, Mesh& mesh)
{
  BilinearForm* a = 0;
  Matrix* A = 0;
  assemble_test(a, &L, *A, b, mesh);
}
//-----------------------------------------------------------------------------
void NewFEM::assemble(BilinearForm& a, LinearForm& L,
		   Matrix& A, Vector& b, Mesh& mesh)
{
  assemble_test(&a, &L, A, b, mesh);
}
//-----------------------------------------------------------------------------
void NewFEM::assemble(BilinearForm& a, LinearForm& L,
		   Matrix& A, Vector& b, Mesh& mesh,
		   BoundaryCondition& bc)
{
  assemble_test(&a, &L, A, b, mesh);
  applyBC(A, b, mesh, a.trial(), bc);
}
//-----------------------------------------------------------------------------
void NewFEM::applyBC(Matrix& A, Vector& b, Mesh& mesh,
		  FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Applying Dirichlet boundary conditions.");

  // Null pointer to a dummy vector
  Vector* x = 0; 

  applyBC(&A, &b, x, mesh, element, bc);
}
//-----------------------------------------------------------------------------
void NewFEM::applyBC(Matrix& A, Mesh& mesh, FiniteElement& element, 
		  BoundaryCondition& bc)
{
  dolfin_info("Applying Dirichlet boundary conditions to matrix.");

  // Null pointer to dummy vectors
  Vector* b = 0; 
  Vector* x = 0; 

  applyBC(&A, b, x, mesh, element, bc);
}
//-----------------------------------------------------------------------------
void NewFEM::applyBC(Vector& b, Mesh& mesh, FiniteElement& element, 
      BoundaryCondition& bc)
{
  dolfin_info("Applying Dirichlet boundary conditions to vector.");
  
  // Null pointer to a matrix and vector
  Matrix* A = 0; 
  Vector* x = 0; 

  applyBC(A, &b, x, mesh, element, bc);
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBCresidual(Vector& b, const Vector& x, Mesh& mesh,
		  FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Assembling boundary condtions into residual vector.");

  // Null pointer to a matrix
  Matrix* A = 0; 

  applyBC(A, &b, &x, mesh, element, bc);
}
//-----------------------------------------------------------------------------
void NewFEM::assembleBCresidual(Matrix& A, Vector& b, const Vector& x, 
      Mesh& mesh, FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Assembling boundary condtions into residual vector.");

  applyBC(&A, &b, &x, mesh, element, bc);
}
//-----------------------------------------------------------------------------
dolfin::uint NewFEM::size(const Mesh& mesh, const FiniteElement& element)
{
  // FIXME: This could be much more efficient. FFC could generate a
  // FIXME: function that calculates the total number of degrees of
  // FIXME: freedom in just a few operations.

  // Count the degrees of freedom (check maximum index)
  
  int* nodes = new int[element.spacedim()];
  int nodemax = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    element.nodemap(nodes, *cell, mesh);
    for (uint i = 0; i < element.spacedim(); i++)
    {
      if ( nodes[i] > nodemax )
        nodemax = nodes[i];
    }
  }
  delete [] nodes;

  return static_cast<uint>(nodemax + 1);
}
//-----------------------------------------------------------------------------
void NewFEM::lump(const Matrix& M, Vector& m)
{
  m.init(M.size(0));

  Vector one(m);
  one = 1.0;

  M.mult(one, m);
}
//-----------------------------------------------------------------------------
void NewFEM::disp(const Mesh& mesh, const FiniteElement& element)
{
  dolfin_info("Assembly data:");
  dolfin_info("--------------");
  dolfin_info("");

  // Total number of nodes
  uint N = size(mesh, element);
  dolfin_info("  Total number of degrees of freedom: %d.", N);
  dolfin_info("");

  // Display mesh data
  mesh.disp();

  // Display finite element data
  element.disp();

  // Allocate local data
  uint n = element.spacedim();
  int* nodes = new int[n];
  uint* components = new uint[n];
  Point* points = new Point[n];
  AffineMap map;

  // Display data for each cell
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    map.update(*cell);
    element.nodemap(nodes, *cell, mesh);
    element.pointmap(points, components, map);

    cout << "  " << *cell << endl;
    
    for (uint i = 0; i < n; i++)
    {
      cout << "    i = " << i << ": x = " << points[i] << " mapped to global node "
	   << nodes[i] << " (component = " << components[i] << ")" << endl;
    }
  }

  // Delete data
  delete [] nodes;
  delete [] components;
  delete [] points;
}
//-----------------------------------------------------------------------------
void NewFEM::assemble_test(BilinearForm* a, LinearForm* L, Matrix& A, Vector& b, 
      Mesh& mesh)
{
  // Check that the mesh matches the forms
  if( a )
    checkdims(*a, mesh);
  if( L )
    checkdims(*L, mesh);
 
  // Get finite elements
  FiniteElement* test_element  = 0;
  FiniteElement* trial_element = 0;
  if( a )
  {
    test_element  = &(a->test());
    trial_element = &(a->trial());
  }
  else if( L ) 
  {
    test_element = &(L->test());
  }  

  // Create affine map
  AffineMap map;

  // Initialize element matrix/vector data block
  real* block_A = 0;
  real* block_b = 0;
  int* test_nodes = 0;
  int* trial_nodes = 0;
  uint n  = 0;
  uint N  = 0;
  uint nz = 0;

  const uint m = test_element->spacedim();
  const uint M = size(mesh, *test_element);
  test_nodes = new int[m];

  if( a )
  {
    n = trial_element->spacedim();
    N = size(mesh, *trial_element);
    block_A = new real[m*n];
    trial_nodes = new int[m];
    nz = nzsize(mesh, *trial_element);
    A.init(M, N, nz);
    A = 0.0;
  }
  if( L )
  {
    block_b = new real[m];  
    b.init(M);
    b = 0.0;
  }
  // Start a progress session
  if( a && L)
    dolfin_info("Assembling system (matrix and vector) of size %d x %d.", M, N);
  if( a && !L)
    dolfin_info("Assembling matrix of size %d x %d.", M, N);
  if( !a && L)
    dolfin_info("Assembling vector of size %d.", M);
  Progress p("Assembling matrix and vector (interior contributions)", mesh.numCells());
 
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update affine map
    map.update(*cell);

    // Compute map from local to global degrees of freedom (test functions)
    test_element->nodemap(test_nodes, *cell, mesh);

    if( a )
    {
      // Update forms
      a->update(map);

      // Compute maps from local to global degrees of freedom (trial functions)
      trial_element->nodemap(trial_nodes, *cell, mesh);
      // Compute element matrix 
      a->eval(block_A, map);

      // Add element matrix to global matrix
      A.add(block_A, test_nodes, m, trial_nodes, n);
    }
    if( L )
    {
      // Update forms
      L->update(map);
    
      // Compute element vector 
      L->eval(block_b, map);

      // Add element vector to global vector
      b.add(block_b, test_nodes, m);
    }

    // Update progress
    p++;
  }
  
  // Complete assembly
  if( L )
    b.apply();
  if ( a )
  {
    A.apply();
    // Check the number of nonzeros
    checknz(A, nz);
  }

  // Delete data
  delete [] block_A;
  delete [] block_b;
  delete [] trial_nodes;
  delete [] test_nodes;

}
//-----------------------------------------------------------------------------
void NewFEM::applyBC(Matrix* A, Vector* b, const Vector* x, Mesh& mesh, 
          FiniteElement& element, BoundaryCondition& bc)
{
  // Create boundary
  Boundary boundary(mesh);

  // Choose type of mesh
  if( mesh.type() == Mesh::triangles)
  {
    BoundaryIterator<EdgeIterator,Edge> boundary_entity(boundary);
    applyBC(A, b, x, mesh, element, bc, boundary_entity);
  }
  else if(mesh.type() == Mesh::tetrahedra)
  {
    BoundaryIterator<FaceIterator,Face> boundary_entity(boundary);
    applyBC(A, b, x, mesh, element, bc, boundary_entity);
  }
  else
  {
    dolfin_error("Unknown mesh type.");  
  }
}
//-----------------------------------------------------------------------------
dolfin::uint NewFEM::nzsize(const Mesh& mesh, const FiniteElement& element)
{
  // Estimate the number of nonzeros in each row

  uint nzmax = 0;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    nzmax = std::max(nzmax, vertex->numVertexNeighbors() * element.spacedim());

  return nzmax;
}
//-----------------------------------------------------------------------------
void NewFEM::checkdims(BilinearForm& a, const Mesh& mesh)
{
  switch ( mesh.type() )
  {
  case Mesh::triangles:
    if ( a.test().shapedim() != 2 )
      dolfin_error("Given mesh (triangular 2D) does not match shape dimension for form.");
    if ( a.trial().shapedim() != 2 )
      dolfin_error("Given mesh (triangular 2D) does not match shape dimension for form.");
    break;
  case Mesh::tetrahedra:
    if ( a.test().shapedim() != 3 )
      dolfin_error("Given mesh (tetrahedral 3D) does not match shape dimension for form.");
    if ( a.trial().shapedim() != 3 )
      dolfin_error("Given mesh (tetrahedral 3D) does not match shape dimension for form.");
    break;
  default:
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void NewFEM::checkdims(LinearForm& L, const Mesh& mesh)
{
  switch ( mesh.type() )
  {
  case Mesh::triangles:
    if ( L.test().shapedim() != 2 )
      dolfin_error("Given mesh (triangular 2D) does not match shape dimension for form.");
    break;
  case Mesh::tetrahedra:
    if ( L.test().shapedim() != 3 )
      dolfin_error("Given mesh (tetrahedral 3D) does not match shape dimension for form.");
    break;
  default:
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void NewFEM::checknz(const Matrix& A, uint nz)
{
  uint nz_actual = A.nzmax();
  if ( nz_actual > nz )
    dolfin_warning("Actual number of nonzero entries exceeds estimated number of nonzero entries.");
  else
    dolfin_info("Maximum number of nonzeros in each row is %d (estimated %d).",
		nz_actual, nz);
}
//-----------------------------------------------------------------------------
