// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Andy Terrel 2005.
// Modified by Garth N. Wells 2005, 2006.
//
// First added:  2004-05-19
// Last changed: 2006-04-25

#include <dolfin/FEM.h>

using namespace dolfin;

#ifdef HAVE_PETSC_H

// FIXME: For testing
void FEM::assembleNoTemplate(BilinearForm& a, GenericMatrixNoTemplate& A, Mesh& mesh)
{
  assembleCommonNoTemplate(&a, 0, &A, 0, mesh);
}

// FIXME: For testing
void FEM::assembleCommonNoTemplate(BilinearForm* a, LinearForm* L, 
				   GenericMatrixNoTemplate* A, GenericVectorNoTemplate* b,
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
    test_element = &(L->test());
  
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
    A->init(M, N, nz);
    *A = 0.0;
  }
  if( L )
  {
    block_b = new real[m];  
    b->init(M);
    *b = 0.0;
  }
  // Start a progress session
  if( a && L)
    dolfin_info("Assembling system (matrix and vector) of size %d x %d.", M, N);
  if( a && !L)
    dolfin_info("Assembling matrix of size %d x %d.", M, N);
  if( !a && L)
    dolfin_info("Assembling vector of size %d.", M);
  Progress p("Assembling interior contributions", mesh.numCells());
  
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
      A->add(block_A, test_nodes, m, trial_nodes, n);
    }
    if( L )
    {
      // Update forms
      L->update(map);    
      // Compute element vector 
      L->eval(block_b, map);
      // Add element vector to global vector
      b->add(block_b, test_nodes, m);
    }
    
    // Update progress
    p++;
  }

  /*
  
  //FIXME: need to reinitiliase block_A and block_b in case no boudary terms are provided
  if( a )
    for (uint i = 0; i < m*n; ++i)
      block_A[i] = 0.0;
  if( L )
    for (uint i = 0; i < m; ++i)
      block_b[i] = 0.0;
  
  // Iterate over all facets on the boundary
  Boundary boundary(mesh);
  Progress p_boundary("Assembling boudary contributions", boundary.numFacets());
  for ( ; !facet.end(); ++facet)
  {
    // Get cell containing the edge (pick first, should only be one)
    dolfin_assert(facet.numCellNeighbors() == 1);
    Cell& cell = facet.cell(0);
    
    // Get local facet ID
    uint facetID = facet.localID(0);
    
    // Update affine map for facet 
    map.update(cell, facetID);
    
    // Compute map from local to global degrees of freedom (test functions)
    test_element->nodemap(test_nodes, cell, mesh);
  
    if( a )
    {
      // Update forms
      a->update(map);  
        // Compute maps from local to global degrees of freedom (trial functions)
      trial_element->nodemap(trial_nodes, cell, mesh);
      
      // Compute element matrix 
      a->eval(block_A, map, facetID);
      
      // Add element matrix to global matrix
      A->add(block_A, test_nodes, m, trial_nodes, n);
    }
    if( L )
    {
      // Update forms
      L->update(map);    
      // Compute element vector 
      L->eval(block_b, map, facetID);
      
      // Add element vector to global vector
      b->add(block_b, test_nodes, m);
    }
    // Update progress
    p_boundary++;
  }

  */
  
  // Complete assembly
  if( L )
    b->apply();
  if ( a )
  {
    A->apply();
    // Check the number of nonzeros
    //checknz(*A, nz);
  }
  
  // Delete data
  delete [] block_A;
  delete [] block_b;
  delete [] trial_nodes;
  delete [] test_nodes;
}















//-----------------------------------------------------------------------------
void FEM::lump(const Matrix& M, Vector& m)
{
  m.init(M.size(0));
  Vector one(m);
  one = 1.0;
  M.mult(one, m);
}
#endif
//-----------------------------------------------------------------------------
  void FEM::lump(const DenseMatrix& M, DenseVector& m)
  {
    m.init(M.size(0));
    DenseVector one(m);
    one = 1.0;
    M.mult(one, m);
  }
//-----------------------------------------------------------------------------
dolfin::uint FEM::size(const Mesh& mesh, const FiniteElement& element)
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
void FEM::disp(const Mesh& mesh, const FiniteElement& element)
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
dolfin::uint FEM::nzsize(const Mesh& mesh, const FiniteElement& element)
{
  // Estimate the number of nonzeros in each row

  uint nzmax = 0;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    nzmax = std::max(nzmax, vertex->numVertexNeighbors() * element.spacedim());

  return nzmax;
}
//-----------------------------------------------------------------------------
void FEM::checkdims(BilinearForm& a, const Mesh& mesh)
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
void FEM::checkdims(LinearForm& L, const Mesh& mesh)
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
