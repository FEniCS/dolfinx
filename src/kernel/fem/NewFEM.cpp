// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Andy Terrel 2005.
// Modified by Garth N. Wells 2005.
//
// First added:  2004-05-19
// Last changed: 2006-04-11

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
  // Check that the mesh matches the form
  checkdims(a, mesh);

  // Get finite elements
  FiniteElement& test_element = a.test();
  FiniteElement& trial_element = a.trial();

  // Create affine map
  AffineMap map;

  // Initialize local data
  const uint m = test_element.spacedim();
  const uint n = trial_element.spacedim();
  real* block = new real[m*n];
  int* test_nodes = new int[m];
  int* trial_nodes = new int[n];

  // Initialize global matrix
  const uint M = size(mesh, test_element);
  const uint N = size(mesh, trial_element);
  const uint nz = nzsize(mesh, trial_element);
  A.init(M, N, nz);
  A = 0.0;

  // Start a progress session
  dolfin_info("Assembling matrix of size %d x %d.", M, N);
  Progress p("Assembling matrix (interior contribution)", mesh.numCells());

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update affine map
    map.update(*cell);

    // Update form
    a.update(map);
    
    // Compute maps from local to global degrees of freedom
    test_element.nodemap(test_nodes, *cell, mesh);
    trial_element.nodemap(trial_nodes, *cell, mesh);


    //tic();
    //for (int i = 0; i < 100000; i++)
    //{
    
    // Compute element matrix
    a.eval(block, map);
        
    //}
    //cout << "Time to assemble: " << toc() << endl;

    // Add element matrix to global matrix
    A.add(block, test_nodes, m, trial_nodes, n);

    // Update progress
    p++;
  }
  
  // Complete assembly
  A.apply();

  // Check number of nonzeros
  checknz(A, nz);

  // Delete data
  delete [] block;
  delete [] test_nodes;
  delete [] trial_nodes;
}
//-----------------------------------------------------------------------------
void NewFEM::assemble(LinearForm& L, Vector& b, Mesh& mesh)
{
  // Check that the mesh matches the form
  checkdims(L, mesh);

  // Get finite element
  FiniteElement& test_element = L.test();

  // Create affine map
  AffineMap map;

  // Initialize local data
  const uint m = test_element.spacedim();
  real* block = new real[m];
  int* test_nodes = new int[m];

  // Initialize global vector 
  const uint M = size(mesh, test_element);
  b.init(M);
  b = 0.0;

  // Start a progress session
  dolfin_info("Assembling vector of size %d.", M);
  Progress p("Assembling vector (interior contribution)", mesh.numCells());

  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update affine map
    map.update(*cell);

    // Update form
    L.update(map);

    // Compute map from local to global degrees of freedom
    test_element.nodemap(test_nodes, *cell, mesh);
    
    // Compute element matrix
    L.eval(block, map);
    
    // Add element matrix to global matrix
    b.add(block, test_nodes, m);

    // Update progress
    p++;
  }
  
  // Complete assembly
  b.apply();

  // Delete data
  delete [] block;
  delete [] test_nodes;
}
//-----------------------------------------------------------------------------
void NewFEM::assemble(BilinearForm& a, LinearForm& L,
		   Matrix& A, Vector& b, Mesh& mesh)
{
  // Check that the mesh matches the forms
  checkdims(a, mesh);
  checkdims(L, mesh);
 
  // Get finite elements
  FiniteElement& test_element = a.test();
  FiniteElement& trial_element = a.trial();

  // Create affine map
  AffineMap map;

  // Initialize element matrix/vector data block
  const uint m = test_element.spacedim();
  const uint n = trial_element.spacedim();
  real* block_A = new real[m*n];
  real* block_b = new real[m];
  int* test_nodes = new int[m];
  int* trial_nodes = new int[n];

  // Initialize global matrix
  const uint M = size(mesh, test_element);
  const uint N = size(mesh, trial_element);
  const uint nz = nzsize(mesh, trial_element);
  A.init(M, N, nz);
  b.init(M);
  A = 0.0;
  b = 0.0;
 
  // Start a progress session
  dolfin_info("Assembling system (matrix and vector) of size %d x %d.", M, N);
  Progress p("Assembling matrix and vector (interior contributions)", mesh.numCells());
 
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update affine map
    map.update(*cell);

    // Update forms
    a.update(map);
    L.update(map);

    // Compute maps from local to global degrees of freedom
    test_element.nodemap(test_nodes, *cell, mesh);
    trial_element.nodemap(trial_nodes, *cell, mesh);
   
    // Compute element matrix and vector 
    a.eval(block_A, map);
    L.eval(block_b, map);
    
    // Add element matrix to global matrix
    A.add(block_A, test_nodes, m, trial_nodes, n);
    
    // Add element vector to global vector
    b.add(block_b, test_nodes, m);

    // Update progress
    p++;
  }
  
  // Complete assembly
  A.apply();
  b.apply();

  // Check the number of nonzeros
  checknz(A, nz);

  // Delete data
  delete [] block_A;
  delete [] block_b;
  delete [] test_nodes;
  delete [] trial_nodes;
}
//-----------------------------------------------------------------------------
void NewFEM::assemble(BilinearForm& a, LinearForm& L,
		   Matrix& A, Vector& b, Mesh& mesh,
		   BoundaryCondition& bc)
{
  assemble(a, L, A, b, mesh);
  applyBC(A, b, mesh, a.trial(), bc);
}
//-----------------------------------------------------------------------------
void NewFEM::applyBC(Matrix& A, Vector& b, Mesh& mesh,
		  FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Applying Dirichlet boundary conditions.");

  enum BCApply type = matrix_and_vector;

  // Create boundary
  Boundary boundary(mesh);

  // Choose type of mesh
  if( mesh.type() == Mesh::triangles )
  {
    BoundaryIterator<EdgeIterator,Edge> boundary_iterator(boundary);
    applyBC(A, b, mesh, element, bc, boundary_iterator, type);
  }
  else if( mesh.type() == Mesh::tetrahedra )
  {
    BoundaryIterator<FaceIterator,Face> boundary_iterator(boundary);
    applyBC(A, b, mesh, element, bc, boundary_iterator, type);
  }
  else
  {
    dolfin_error("Unknown mesh type.");  
  }
}
//-----------------------------------------------------------------------------
void NewFEM::applyBC(Matrix& A, Mesh& mesh, FiniteElement& element, 
		  BoundaryCondition& bc)
{
  dolfin_info("Applying Dirichlet boundary conditions to matrix.");

  Vector b; // Dummy vector
  enum BCApply type = matrix_only;

  // Create boundary
  Boundary boundary(mesh);

  // Choose type of mesh
  if( mesh.type() == Mesh::triangles )
  {
    BoundaryIterator<EdgeIterator,Edge> boundary_entity(boundary);
    applyBC(A, b, mesh, element, bc, boundary_entity, type);
  }
  else if( mesh.type() == Mesh::tetrahedra )
  {
    BoundaryIterator<FaceIterator,Face> boundary_entity(boundary);
    applyBC(A, b, mesh, element, bc, boundary_entity, type);
  }
  else
  {
    dolfin_error("Unknown mesh type.");  
  }

}
//-----------------------------------------------------------------------------
void NewFEM::applyBC(Vector& b, Mesh& mesh, FiniteElement& element, 
      BoundaryCondition& bc)
{
  dolfin_info("Applying Dirichlet boundary conditions to vector.");

  Matrix A; // Dummy matrix
  enum BCApply type = vector_only;

  // Create boundary
  Boundary boundary(mesh);

  // Choose type of mesh
  if( mesh.type() == Mesh::triangles)
  {
    BoundaryIterator<EdgeIterator,Edge> boundary_entity(boundary);
    applyBC(A, b, mesh, element, bc, boundary_entity, type);
  }
  else if(mesh.type() == Mesh::tetrahedra)
  {
    BoundaryIterator<FaceIterator,Face> boundary_entity(boundary);
    applyBC(A, b, mesh, element, bc, boundary_entity, type);
  }
  else
  {
    dolfin_error("Unknown mesh type.");  
  }

}
//-----------------------------------------------------------------------------
void NewFEM::assembleBCresidual(Vector& b, const Vector& x, Mesh& mesh,
		  FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Assembling boundary condtions into residual vector.");

  // Create boundary
  Boundary boundary(mesh);

  // Choose type of mesh
  if( mesh.type() == Mesh::triangles)
  {
    BoundaryIterator<EdgeIterator,Edge> boundary_entity(boundary);
    assembleBCresidual(b, x, mesh, element, bc, boundary_entity);
  }
  else if(mesh.type() == Mesh::tetrahedra)
  {
    BoundaryIterator<FaceIterator,Face> boundary_entity(boundary);
    assembleBCresidual(b, x, mesh, element, bc, boundary_entity);
  }
  else
  {
    dolfin_error("Unknown mesh type.");  
  }
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
