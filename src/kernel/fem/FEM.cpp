// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Andy Terrel 2005.
// Modified by Garth N. Wells 2005, 2006.
//
// First added:  2004-05-19
// Last changed: 2006-09-18

#include <dolfin/Facet.h>
#include <dolfin/FEM.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void FEM::assemble(BilinearForm& a, LinearForm& L,
		   GenericMatrix& A, GenericVector& b,
		   Mesh& mesh)
{
  assembleCommon(&a, &L, 0, &A, &b, 0, mesh);
}
//-----------------------------------------------------------------------------
void FEM::assemble(BilinearForm& a, LinearForm& L,
		   GenericMatrix& A, GenericVector& b,
		   Mesh& mesh, BoundaryCondition& bc)
{
  assembleCommon(&a, &L, 0, &A, &b, 0, mesh);
  applyBC(A, b, mesh, a.trial(), bc);
}
//-----------------------------------------------------------------------------
void FEM::assemble(BilinearForm& a, GenericMatrix& A, Mesh& mesh)
{
  assembleCommon(&a, 0, 0, &A, 0, 0, mesh);
}
//-----------------------------------------------------------------------------
void FEM::assemble(LinearForm& L, GenericVector& b, Mesh& mesh)
{
  assembleCommon(0, &L, 0, (DenseMatrix*) 0, &b, 0, mesh);
}
//-----------------------------------------------------------------------------
real FEM::assemble(Functional& M, Mesh& mesh)
{
  real val = 0.0;
  assembleCommon(0, 0, &M, 0, 0, &val, mesh);
  return val;
}
//-----------------------------------------------------------------------------
void FEM::applyBC(GenericMatrix& A, GenericVector& b,
		  Mesh& mesh, FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Applying Dirichlet boundary conditions.");
  applyCommonBC(&A, &b, 0, mesh, element, bc);
}
//-----------------------------------------------------------------------------
void FEM::applyBC(GenericMatrix& A, Mesh& mesh,
		  FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Applying Dirichlet boundary conditions to matrix.");
  applyCommonBC(&A, (Vector*) 0, 0, mesh, element, bc);
}
//-----------------------------------------------------------------------------
void FEM::applyBC(GenericVector& b, Mesh& mesh,
		  FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Applying Dirichlet boundary conditions to vector.");
  applyCommonBC(0, &b, 0, mesh, element, bc);
}
//-----------------------------------------------------------------------------
void FEM::assembleResidualBC(GenericMatrix& A, GenericVector& b,
			     const GenericVector& x, Mesh& mesh,
			     FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Assembling boundary condtions into residual vector.");
  applyCommonBC(&A, &b, &x, mesh, element, bc);
}
//-----------------------------------------------------------------------------
void FEM::assembleResidualBC(GenericVector& b,
			     const GenericVector& x, Mesh& mesh,
			     FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Assembling boundary condtions into residual vector.");
  applyCommonBC((DenseMatrix*) 0, &b, &x, mesh, element, bc);
}
//-----------------------------------------------------------------------------
dolfin::uint FEM::size(Mesh& mesh, const FiniteElement& element)
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
void FEM::disp(Mesh& mesh, const FiniteElement& element)
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
void FEM::assembleCommon(BilinearForm* a, LinearForm* L, Functional* M,
			 GenericMatrix* A, GenericVector* b, real* val,
			 Mesh& mesh)
{
  // Check that the mesh matches the forms
  if ( a )
    checkDimensions(*a, mesh);
  if ( L )
    checkDimensions(*L, mesh);
  // FIXME: Add dimension check for M
  
  // Create affine map
  AffineMap map;
  
  // Initialize element matrix/vector data block
  uint nz = 0;
  bool interior_contribution = false;
  bool boundary_contribution = false;
  if ( a )
  { 
    const uint M = size(mesh, a->test());
    const uint N = size(mesh, a->trial());
    nz = estimateNonZeros(mesh, a->trial());
    A->init(M, N, nz);
    A->zero();
    interior_contribution = interior_contribution || a->interior_contribution();
    boundary_contribution = boundary_contribution || a->boundary_contribution();
    dolfin_info("Assembling system (matrix and vector) of size %d x %d.", M, N);
  }
  if ( L )
  {
    const uint M = size(mesh, L->test());
    b->init(M);
    b->zero();
    interior_contribution = interior_contribution || L->interior_contribution();
    boundary_contribution = boundary_contribution || L->boundary_contribution();
    dolfin_info("Assembling vector of size %d.", M);
  }
  if ( M )
  {
    *val = 0.0;
    interior_contribution = interior_contribution || M->interior_contribution();
    boundary_contribution = boundary_contribution || M->boundary_contribution();
    dolfin_info("Assembling functional.");
  }
  
  // Assemble interior contribution
  if ( interior_contribution )
  {
    // Start a progress session
    Progress p("Assembling interior contributions", mesh.numCells());
    
    // Iterate over all cells in the mesh
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Update affine map
      map.update(*cell);
      
      // Assemble bilinear form
      if ( a )
	if ( a->interior_contribution() )
	  assembleElement(*a, *A, mesh, *cell, map, -1);              
      
      // Assemble linear form
      if ( L )
	if ( L->interior_contribution() )
	  assembleElement(*L, *b, mesh, *cell, map, -1);              
      
      // Assemble functional
      if ( M )
	if ( M->interior_contribution() )
	  assembleElement(*M, *val, map, -1);              
      
      // Update progress
      p++;
    }
  }
  
  // Assemble boundary contribution
  if ( boundary_contribution )
  {
    // Iterate over all cells in the boundary mesh
    MeshFunction<uint> vertex_map;
    MeshFunction<uint> cell_map;
    BoundaryMesh boundary(mesh, vertex_map, cell_map);
    Progress p_boundary("Assembling boundary contributions",
			boundary.numCells());
    for (CellIterator cell(boundary); !cell.end(); ++cell)
    {
      // Get mesh cell containing the boundary cell (pick first, should only be one)
      Facet facet(mesh, cell_map(*cell));
      dolfin_assert(facet.numConnections(mesh.topology().dim()) == 1);
      Cell mesh_cell(mesh, facet.connections(mesh.topology().dim())[0]);
      
      dolfin_error("localID not implemented");
      
      // Get local facet ID
      //         uint facetID = facet.localID(0);
      uint facetID = 0;
      
      // Update affine map for facet 
      map.update(mesh_cell, facetID);
      
      // Assemble bilinear form
      if ( a )
	if ( a->boundary_contribution() )
	  assembleElement(*a, *A, mesh, mesh_cell, map, facetID);              
      
      // Assemble linear form
      if ( L )
	if ( L->boundary_contribution() )
	  assembleElement(*L, *b, mesh, mesh_cell, map, facetID);              
      
      // Assemble functional
      if ( M )
	if ( M->boundary_contribution() )
	  assembleElement(*M, *val, map, facetID);              
      
      // Update progress  
      p_boundary++;
    }
  }
  
  // Complete assembly
  if ( a )
  {
    A->apply();
    countNonZeros(*A, nz);
  }
  if ( L )
    b->apply();
  
}
//-----------------------------------------------------------------------------
void FEM::assembleCommonOld(BilinearForm* a, LinearForm* L, Functional* M,
			 GenericMatrix* A, GenericVector* b, real* val,
			 Mesh& mesh)
{
  // Create boundary
  BoundaryMesh boundary(mesh);
  
  assembleCommon(a, L, M, A, b, val, mesh);
}
//-----------------------------------------------------------------------------
void FEM::applyCommonBC(GenericMatrix* A, GenericVector* b, 
			const GenericVector* x, Mesh& mesh,
			FiniteElement& element, BoundaryCondition& bc)
{
  // Create boundary value
  BoundaryValue bv;
  
  // Create affine map
  AffineMap map;
  
  // Compute problem size
  uint size = 0;
  if( A )
    size = A->size(0);
  else
    size = b->size();
  
  // Allocate list of rows
  uint m = 0;
  int*  rows = 0;
  if ( A )
    rows = new int[size];
  
  bool* row_set = new bool[size];
  for (unsigned int i = 0; i < size; i++)
    row_set[i] = false;
  
  // Allocate local data
  uint n = element.spacedim();
  int* nodes = new int[n];
  uint* components = new uint[n];
  Point* points = new Point[n];
  
  real* block_b = 0;
  int* node_list = 0;
  if( b )
  {
    block_b   = new real[n];  
    node_list = new int[n];  
  }
  
  // FIXME: Creating a new BoundaryMesh is likely inefficient
  // Iterate over all cells in the boundary mesh
  MeshFunction<uint> vertex_map;
  MeshFunction<uint> cell_map;
  BoundaryMesh boundary(mesh, vertex_map, cell_map);

  // FIXME: Boundary mesh needs to include connected cells in the
  // interior.
//   dolfin_error("Boundary conditions not implemented.");
  
  // Iterate over all cells in the boundary mesh
  for (CellIterator facet(boundary); !facet.end(); ++facet)
  {
    uint k = 0;
    
    // Get cell containing the edge (pick first, should only be one)
    //     dolfin_assert(facet.numConnections(mesh.topology().dim()) == 1);
    //     Cell cell(mesh, (*facet).connections(mesh.topology().dim())[0]);
    Cell cell(mesh, cell_map(*facet));
    
    // Update affine map
    map.update(cell);
    // Compute map from local to global degrees of freedom
    element.nodemap(nodes, cell, mesh);
    // Compute map from local to global coordinates
    element.pointmap(points, components, map);
    
    // Set boundary conditions for nodes on the boundary
    for (uint i = 0; i < n; i++)
    {
      dolfin_warning("contains() not implemented");
      
      // Skip points that are not contained in facet
      const Point& point = points[i];
      //         if ( !(facet.contains(point)) )
      //           continue;
      
      // Get boundary condition
      bv.reset();
      bc.eval(bv, point, components[i]);
      
      // Set boundary condition if Dirichlet
      if ( bv.fixed )
      {
	int node = nodes[i];
	if ( !row_set[node] )
	{
	  if ( x ) // Compute "residual" 
	    block_b[k] = bv.value - x->get(node);
	  else if ( b ) 
	    block_b[k] = bv.value;
	  
	  row_set[node] = true;
	  
	  if ( b )
	    node_list[k++] = node;
	  if ( A )
	    rows[m] = node;
	  
	  m++;
	}
      }
    }
    if( b )
      b->set(block_b, node_list, k);
  }
  dolfin_info("Boundary condition applied to %d degrees of freedom on the boundary.", m);
  
  // Set rows of matrix to the identity matrix
  if( A )
    A->ident(rows, m);
  
  if( b )
    b->apply();
  
  // Delete data
  delete [] nodes;
  delete [] components;
  delete [] points;
  delete [] rows;
  delete [] row_set;
  delete [] block_b;
  delete [] node_list;
}
//-----------------------------------------------------------------------------
void FEM::applyCommonBCOld(GenericMatrix* A, GenericVector* b, 
			const GenericVector* x, Mesh& mesh,
			FiniteElement& element, BoundaryCondition& bc)
{
  // Create boundary
  BoundaryMesh boundary(mesh);
  
  applyCommonBC(A, b, x, mesh, element, bc);
}
//-----------------------------------------------------------------------------
void FEM::checkDimensions(const BilinearForm& a, const Mesh& mesh)
{
  switch ( mesh.topology().dim() )
  {
  case 2:
    if ( a.test().shapedim() != 2 || a.trial().shapedim() != 2 )
      dolfin_error("Given mesh (triangular 2D) does not match shape dimension for form.");
    break;
  case 3:
    if ( a.test().shapedim() != 3 || a.trial().shapedim() != 3 )
      dolfin_error("Given mesh (tetrahedral 3D) does not match shape dimension for form.");
    break;
  default:
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void FEM::checkDimensions(const LinearForm& L, const Mesh& mesh)
{
  switch ( mesh.topology().dim() )
  {
  case 2:
    if ( L.test().shapedim() != 2 )
      dolfin_error("Given mesh (triangular 2D) does not match shape dimension for form.");
    break;
  case 3:
    if ( L.test().shapedim() != 3 )
      dolfin_error("Given mesh (tetrahedral 3D) does not match shape dimension for form.");
    break;
  default:
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
dolfin::uint FEM::estimateNonZeros(Mesh& mesh,
				   const FiniteElement& element)
{
  uint nzmax = 0;

  mesh.init(0, 0);

  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    nzmax = std::max(nzmax, vertex->numConnections(0)*element.spacedim());
  }

  return nzmax;
}
//-----------------------------------------------------------------------------
void FEM::countNonZeros(const GenericMatrix& A, uint nz)
{
  uint nz_actual = A.nzmax();
  if ( nz_actual > nz )
    dolfin_warning("Actual number of nonzero entries exceeds estimated number of nonzero entries.");
  else
    dolfin_info("Maximum number of nonzeros in each row is %d (estimated %d).",
		nz_actual, nz);
}
//-----------------------------------------------------------------------------
