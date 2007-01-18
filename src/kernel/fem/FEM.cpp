// Copyright (C) 2004-2007 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Andy Terrel 2005.
// Modified by Garth N. Wells 2005, 2006.
//
// First added:  2004-05-19
// Last changed: 2007-01-16

#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/Functional.h>
#include <dolfin/Mesh.h>
#include <dolfin/Cell.h>
#include <dolfin/Facet.h>
#include <dolfin/Point.h>
#include <dolfin/MeshFunction.h>
#include <dolfin/BoundaryMesh.h>
#include <dolfin/GenericMatrix.h>
#include <dolfin/GenericVector.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/BoundaryValue.h>
#include <dolfin/AffineMap.h>
#include <dolfin/Interval.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/DofMap.h>
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
  assembleCommon(0, &L, 0, 0, &b, 0, mesh);
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
  applyCommonBC(&A, 0, 0, mesh, element, bc);
}
//-----------------------------------------------------------------------------
void FEM::applyBC(GenericVector& b, Mesh& mesh,
		  FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Applying Dirichlet boundary conditions to vector.");
  applyCommonBC(0, &b, 0, mesh, element, bc);
}
//-----------------------------------------------------------------------------
void FEM::applyResidualBC(GenericMatrix& A, GenericVector& b,
			     const GenericVector& x, Mesh& mesh,
			     FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Applying boundary conditions to residual vector.");
  applyCommonBC(&A, &b, &x, mesh, element, bc);
}
//-----------------------------------------------------------------------------
void FEM::applyResidualBC(GenericVector& b,
			     const GenericVector& x, Mesh& mesh,
			     FiniteElement& element, BoundaryCondition& bc)
{
  dolfin_info("Assembling boundary conditions to residual vector.");
  applyCommonBC(0, &b, &x, mesh, element, bc);
}
//-----------------------------------------------------------------------------
dolfin::uint FEM::size(Mesh& mesh, const FiniteElement& element)
{
  // FIXME: This could be much more efficient. FFC could generate a
  // FIXME: function that calculates the total number of degrees of
  // FIXME: freedom in just a few operations.

  /// FIXME: This function is used by Functions, but the size should be computed
  /// via DofMap. It might be removed. 

  // Initialize connectivity
  initConnectivity(mesh);

  // Count the degrees of freedom (check maximum index)
  int* dofs = new int[element.spacedim()];
  int dofmax = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    element.nodemap(dofs, *cell, mesh);
    for (uint i = 0; i < element.spacedim(); i++)
      dofmax = std::max(dofs[i], dofmax);
  }
  delete [] dofs;

  return static_cast<uint>(dofmax + 1);
}
//-----------------------------------------------------------------------------
void FEM::disp(Mesh& mesh, const FiniteElement& element)
{
  dolfin_info("Assembly data:");
  dolfin_info("--------------");
  dolfin_info("");

  // Initialize connectivity
  initConnectivity(mesh);

  // Total number of degrees of freedom
  DofMap dofmap(mesh, &element);  
  uint N = dofmap.size();
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
  if ( a ) checkDimensions(*a, mesh);
  if ( L ) checkDimensions(*L, mesh);
  // FIXME: Add dimension check for M

  // Initialize connectivity
  initConnectivity(mesh);

  // Create affine map
  AffineMap map;
  
  // Create degree of freedom maps
  // FIXME: DofMap should be computed only once unless mesh/element changes
  DofMap dofmap_a(mesh);
  DofMap dofmap_L(mesh);

  // Initialize global data
  bool interior_contribution = false;
  bool boundary_contribution = false;
  bool interior_boundary_contribution = false;
  
  if ( a )
  { 
    dofmap_a.attach(&(a->test()), &(a->trial()));
    const uint M = dofmap_a.size(0);
    const uint N = dofmap_a.size(1);;
    int* nzrow = new int[M];
    dofmap_a.numNonZeroesRow(nzrow);
    A->init(M, N, nzrow);
    A->zero();
    delete [] nzrow;    

    interior_contribution = interior_contribution || a->interior_contribution();
    boundary_contribution = boundary_contribution || a->boundary_contribution();
    interior_boundary_contribution = interior_boundary_contribution || a->interior_boundary_contribution();
    dolfin_info("Assembling system (matrix and vector) of size %d x %d.", M, N);
  }
  if ( L )
  {
    dofmap_L.attach(&(L->test()));
    const uint M = dofmap_L.size();
    b->init(M);
    b->zero();

    interior_contribution = interior_contribution || L->interior_contribution();
    boundary_contribution = boundary_contribution || L->boundary_contribution();
    //interior_boundary_contribution = interior_boundary_contribution || L->interior_boundary_contribution();
    dolfin_info("Assembling vector of size %d.", M);
  }
  if ( M )
  {
    *val = 0.0;
    interior_contribution = interior_contribution || M->interior_contribution();
    boundary_contribution = boundary_contribution || M->boundary_contribution();
    //interior_boundary_contribution = interior_boundary_contribution || M->interior_boundary_contribution();
    dolfin_info("Assembling functional.");
  }
  
  // Assemble interior contribution
  if ( interior_contribution )
  {
    // Iterate over all cells in the mesh
    Progress p("Assembling interior contributions", mesh.numCells());
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Update affine map
      map.update(*cell);

      // Get determinant for integral volume change
      real det = map.det;
      
      // Assemble bilinear form
      if ( a && a->interior_contribution() )
        assembleElementTensor(*a, *A, mesh, *cell, map, det, dofmap_a);
      
      // Assemble linear form
      if ( L && L->interior_contribution() )
        assembleElementTensor(*L, *b, mesh, *cell, map, det, dofmap_L);
      
      // Assemble functional
      if ( M && M->interior_contribution() )
        assembleElementTensor(*M, *val, *cell, map, det);
      
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
    Progress p("Assembling boundary contributions", boundary.numCells());
    for (CellIterator boundary_cell(boundary); !boundary_cell.end(); ++boundary_cell)
    {
      // Create mesh facet corresponding to boundary cell
      Facet mesh_facet(mesh, cell_map(*boundary_cell));
      dolfin_assert(mesh_facet.numEntities(mesh.topology().dim()) == 1);

      // Get cell to which facet belongs (pick first, there is only one)
      Cell mesh_cell(mesh, mesh_facet.entities(mesh.topology().dim())[0]);

      // Get local index of facet with respect to the cell
      uint local_facet = mesh_cell.index(mesh_facet);
      
      // Update affine map for facet 
      map.update(mesh_cell);
      
      // Compute determinant for integral volume change
      real det = computeDeterminant(*boundary_cell);
      
      // Assemble bilinear form
      if ( a && a->boundary_contribution() )
        assembleExteriorFacetTensor(*a, *A, mesh, mesh_cell, map, det, local_facet);
      
      // Assemble linear form
      if ( L && L->boundary_contribution() )
        assembleExteriorFacetTensor(*L, *b, mesh, mesh_cell, map, det, local_facet);
      
      // Assemble functional
      if ( M && M->boundary_contribution() )
        assembleExteriorFacetTensor(*M, *val, mesh_cell, map, det, local_facet);
      
      // Update progress  
      p++;
    }
  }

  // Assemble interior boundary contribution
  if ( interior_boundary_contribution )
  {
    // Make sure the connectivity facet - cell is computed
    mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());

    // We need two affine maps
    AffineMap map0, map1;

    // Iterate over all facets in the mesh
    Progress p("Assembling interior boundary contributions", mesh.numFacets());
    for (FacetIterator facet(mesh); !facet.end(); ++facet)
    {
      // Check if we have an interior facet
      if ( facet->numEntities(mesh.topology().dim()) != 2 )
      {
        p++;
        continue;
      }

      // Get cells associated with facet
      Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
      Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);
      
      // Get local index of facet with respect to each cell
      uint facet0 = cell0.index(*facet);
      uint facet1 = cell1.index(*facet);

      // Compute alignment
      uint alignment = computeAlignment(cell0, cell1, facet->index());

      // Update affine maps for cells
      map0.update(cell0);
      map1.update(cell1);

      // Compute determinant for integral volume change
      real det = computeDeterminant(*facet);
      
      // Assemble bilinear form
      if ( a && a->interior_boundary_contribution() )
        assembleInteriorFacetTensor(*a, *A, mesh, cell0, cell1, map0, map1, det, facet0, facet1, alignment);
      
      // Assemble linear form
      if ( L && L->interior_boundary_contribution() )
        assembleInteriorFacetTensor(*L, *b, mesh, cell0, cell1, map0, map1, det, facet0, facet1, alignment);
      
      // Assemble functional
      if ( M && M->interior_boundary_contribution() )
        assembleInteriorFacetTensor(*M, *val, cell0, cell1, map0, map1, det, facet0, facet1, alignment);
      
      // Update progress  
      p++;
    }
  }
  
  // Complete assembly
  if ( a )
    A->apply();
  if ( L )
    b->apply();
}
//-----------------------------------------------------------------------------
void FEM::applyCommonBC(GenericMatrix* A, GenericVector* b, 
			const GenericVector* x, Mesh& mesh,
			FiniteElement& element, BoundaryCondition& bc)
{
  // Initialize connectivity
  initConnectivity(mesh);

  // Create boundary value
  BoundaryValue bv;
  
  // Create affine map
  AffineMap map;
  
  // Compute problem size
  uint size = 0;
  if ( A )
    size = A->size(0);
  else
    size = b->size();
  
  // Allocate list of rows
  uint m = 0;
  int* rows = 0;
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
  
  // FIXME: Creating a new BoundaryMesh every time is likely inefficient
  
  // Iterate over all cells in the boundary mesh
  MeshFunction<uint> vertex_map;
  MeshFunction<uint> cell_map;
  BoundaryMesh boundary(mesh, vertex_map, cell_map);

  // FIXME: What does this comment mean? I put it here but don't remember why.
  // FIXME: Boundary mesh needs to include connected cells in the interior.
  
  // Iterate over all cells in the boundary mesh
  for (CellIterator boundary_cell(boundary); !boundary_cell.end(); ++boundary_cell)
  {
    uint k = 0;

    // Create mesh facet corresponding to boundary cell
    Facet mesh_facet(mesh, cell_map(*boundary_cell));
    dolfin_assert(mesh_facet.numEntities(mesh.topology().dim()) == 1);
    
    // Get cell to which facet belongs (pick first, there is only one)
    Cell mesh_cell(mesh, mesh_facet.entities(mesh.topology().dim())[0]);
    
    // Update affine map
    map.update(mesh_cell);
    
    // Compute map from local to global degrees of freedom
    element.nodemap(nodes, mesh_cell, mesh);
    
    // Compute map from local to global coordinates
    element.pointmap(points, components, map);
    
    // Set boundary conditions for nodes on the boundary
    for (uint i = 0; i < n; i++)
    {
      // Skip interior points
      const Point& point = points[i];
      if ( !onFacet(point, *boundary_cell) )
        continue;
      
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
void FEM::assembleElementTensor(BilinearForm& a, GenericMatrix& A, 
                                const Mesh& mesh, Cell& cell,
                                AffineMap& map, real det, const DofMap& dofmap)
{
  // Update form
  a.update(cell, map);
  
  // Compute maps from local to global degrees of freedom
  dofmap.dofmap(a.test_nodes, cell, 0);
  dofmap.dofmap(a.trial_nodes, cell, 1);

  // Compute element tensor
  a.eval(a.block, map, det);
  
  // Add element tensor to global tensor
  A.add(a.block, a.test_nodes, a.test().spacedim(), a.trial_nodes, a.trial().spacedim());
}
//-----------------------------------------------------------------------------
void FEM::assembleElementTensor(LinearForm& L, GenericVector& b, 
                                const Mesh& mesh, Cell& cell,
                                AffineMap& map, real det, const DofMap& dofmap)
{
  // Update form
  L.update(cell, map);
  
  // Compute map from local to global degrees of freedom
  dofmap.dofmap(L.test_nodes, cell);
  
  // Compute element tensor
  L.eval(L.block, map, det);
  
  // Add element tensor to global tensor
  b.add(L.block, L.test_nodes, L.test().spacedim());
}
//-----------------------------------------------------------------------------
void FEM::assembleElementTensor(Functional& M, real& val, Cell& cell,
                                AffineMap& map, real det)
{
  // Update form
  M.update(cell, map);
  
  // Compute element tensor
  M.eval(M.block, map, det);
  
  // Add element tensor to global tensor
  val += M.block[0];
}
//-----------------------------------------------------------------------------
void FEM::assembleExteriorFacetTensor(BilinearForm& a, GenericMatrix& A, 
                                      const Mesh& mesh, Cell& cell,
                                      AffineMap& map, real det, uint facet)
{
  // Update form
  a.update(cell, map, facet);
  
  // Compute maps from local to global degrees of freedom
  a.test().nodemap(a.test_nodes, cell, mesh);
  a.trial().nodemap(a.trial_nodes, cell, mesh);
  
  // Compute exterior facet tensor
  a.eval(a.block, map, det, facet);
  
  // Add exterior facet tensor to to global tensor
  A.add(a.block, a.test_nodes, a.test().spacedim(), a.trial_nodes, a.trial().spacedim());
}
//-----------------------------------------------------------------------------
void FEM::assembleExteriorFacetTensor(LinearForm& L, GenericVector& b, 
                                      const Mesh& mesh, Cell& cell,
                                      AffineMap& map, real det, uint facet)
{
  // Update form
  L.update(cell, map);
  
  // Compute map from local to global degrees of freedom
  L.test().nodemap(L.test_nodes, cell, mesh);
  
  // Compute exterior facet tensor
  L.eval(L.block, map, det, facet);
  
  // Add exterior facet tensor to global tensor
  b.add(L.block, L.test_nodes, L.test().spacedim());
}
//-----------------------------------------------------------------------------
void FEM::assembleExteriorFacetTensor(Functional& M, real& val, Cell& cell,
                                      AffineMap& map, real det, uint facet)
{
  // Update form
  M.update(cell, map);
  
  // Compute exterior facet tensor
  M.eval(M.block, map, det, facet);
  
  // Add exterior facet tensor to global tensor
  val += M.block[0];
}
//-----------------------------------------------------------------------------
void FEM::assembleInteriorFacetTensor(BilinearForm& a, GenericMatrix& A, 
                                      const Mesh& mesh,
                                      Cell& cell0, Cell& cell1,
                                      AffineMap& map0, AffineMap& map1, real det,
                                      uint facet0, uint facet1, uint alignment)
{
  // Update form
  a.update(cell0, cell1, map0, map1, facet0, facet1);
  
  cout << "(facet0, facet1, alignment, det)" << "("<<facet0<<", "<<facet1<<", "<<alignment<<", "<<det<< ")"<<endl;

  // Initialize local data structures (FIXME: reuse)
  const uint m = a.test().spacedim();
  const uint n = a.trial().spacedim();
  real* block = new real[4*m*n];
  int* test_nodes = new int[2*m];
  int* trial_nodes = new int[2*n];
  
  // Compute dof map for cell 0 (+ side)
  a.test().nodemap(a.test_nodes, cell0, mesh);
  a.trial().nodemap(a.trial_nodes, cell0, mesh);

  // Copy dofs to common array
  for (uint i = 0; i < m; i++)
    test_nodes[i] = a.test_nodes[i];
  for (uint i = 0; i < n; i++)
    trial_nodes[i] = a.trial_nodes[i];

  // Compute dof map for cell 1 (- side)
  a.test().nodemap(a.test_nodes, cell1, mesh);
  a.trial().nodemap(a.trial_nodes, cell1, mesh);

  // Copy dofs to common array
  for (uint i = 0; i < m; i++)
    test_nodes[m+i] = a.test_nodes[i];
  for (uint i = 0; i < n; i++)
    trial_nodes[n+i] = a.trial_nodes[i];

  // Compute interior facet tensor
  a.eval(block, map0, map1, det, facet0, facet1, alignment);

  for (int i(0); i<8; i++)
  {
    for (int j(0); j<8; j++)
      cout << block[i*8+j] << " ";
    cout <<"\n";
  }
  cout << endl;

  // Add exterior facet tensor to global tensor
  A.add(block, test_nodes, 2*m, trial_nodes, 2*n);

  // Delete local data structures
  delete [] block;
  delete [] test_nodes;
  delete [] trial_nodes;
}
//-----------------------------------------------------------------------------
void FEM::assembleInteriorFacetTensor(LinearForm& L, GenericVector& b, 
                                      const Mesh& mesh,
                                      Cell& cell0, Cell& cell1,
                                      AffineMap& map0, AffineMap& map1, real det,
                                      uint facet0, uint facet1, uint alignment)
{
  // Update form
  L.update(cell0, cell1, map0, map1, facet0, facet1);

  // Initialize local data structures (FIXME: reuse)
  const uint m = L.test().spacedim();
  real* block = new real[2*m];
  int* test_nodes = new int[2*m];
  
  // Compute dof map for cell 0 (+ side)
  L.test().nodemap(L.test_nodes, cell0, mesh);

  // Copy dofs to common array
  for (uint i = 0; i < m; i++)
    test_nodes[i] = L.test_nodes[i];

  // Compute dof map for cell 1 (- side)
  L.test().nodemap(L.test_nodes, cell1, mesh);

  // Copy dofs to common array
  for (uint i = 0; i < m; i++)
    test_nodes[m+i] = L.test_nodes[i];

  // Compute interior facet tensor
  L.eval(block, map0, map1, det, facet0, facet1, alignment);

  // Add exterior facet tensor to global tensor
  b.add(block, test_nodes, 2*m);

  // Delete local data structures
  delete [] block;
  delete [] test_nodes;
}
//-----------------------------------------------------------------------------
void FEM::assembleInteriorFacetTensor(Functional& M, real& val,
                                      Cell& cell0, Cell& cell1,
                                      AffineMap& map0, AffineMap& map1, real det,
                                      uint facet0, uint facet1, uint alignment)
{
  // Update form
  M.update(cell0, cell1, map0, map1, facet0, facet1);
  
  // Compute interior facet tensor
  M.eval(M.block, map0, map1, det, facet0, facet1, alignment);

  // Add exterior facet tensor to global tensor
  val += M.block[0];
}
//-----------------------------------------------------------------------------
void FEM::initConnectivity(Mesh& mesh)
{
  // This is a temporary fix. We need to get information from FFC about
  // which connectivity is needed for the mapping of nodes.

  // This is needed to for the mapping of nodes
  for (uint i = 0; i < mesh.topology().dim(); i++)
    mesh.init(i);

  // This is needed for higher order Lagrange elements (degree >= 4)
  // to compute the alignment of faces
  if ( mesh.topology().dim() == 3 )
    mesh.init(2, 1);
}
//-----------------------------------------------------------------------------
bool FEM::onFacet(const Point& p, Cell& facet)
{
  // Get mesh geometry and vertices of facet
  const MeshGeometry& geometry = facet.mesh().geometry();
  const uint* vertices = facet.entities(0);

  if ( facet.dim() == 1 )
  {
    // Check if point is on the same line as the line segment
    Point v0  = geometry.point(vertices[0]);
    Point v1  = geometry.point(vertices[1]);
    Point v01 = v1 - v0;
    Point v0p = p - v0;
    return v01.cross(v0p).norm() < DOLFIN_EPS;
  }
  else if ( facet.dim() == 2 )
  {
    // Check if point is in the same plane as the triangular facet
    Point v0  = geometry.point(vertices[0]);
    Point v1  = geometry.point(vertices[1]);
    Point v2  = geometry.point(vertices[2]);
    Point v01 = v1 - v0;
    Point v02 = v2 - v0;
    Point v0p = p - v0;
    Point n   = v01.cross(v02);
    return std::abs(n.dot(v0p)) < DOLFIN_EPS;
  }
  
  dolfin_error("Unable to determine if given point is on facet.");
  return false;
}
//-----------------------------------------------------------------------------
real FEM::computeDeterminant(MeshEntity& facet)
{
  // FIXME: Not sure the scaling is correct, need to check
  // FIXME: on the FFC side

  Interval interval;
  Triangle triangle;

  switch ( facet.dim() )
  {
  case 1:
    // Length of reference interval is 1
    return interval.volume(facet);
    break;
  case 2:
    // Area of reference triangle is 0.5
    return triangle.volume(facet) / 0.5;
    break;
  default:
    dolfin_error("Unknown cell type for facet determinant.");
  }

  return 0.0;
}
//-----------------------------------------------------------------------------
dolfin::uint FEM::computeAlignment(Cell& cell0, Cell& cell1, uint facet)
{
  // Create facet.
  Facet f(cell0.mesh(), facet);


  if (f.dim() != 1 && f.dim() != 2)
    dolfin_error("Alignment is currently only defined for a line or triangle");

  // Currently only the alignment of a line with respect to a triangle is implemented
  if ( f.dim() == 1 )
  {
    // Mapping from pair of global alignments to alignment of facet pair
    static unsigned int pair_alignments[2][2] = {{0, 1},
                                                  {1, 0}};

    // Get local index of facet with respect to each cell.
    uint facet0 = cell0.index(f);
    uint facet1 = cell1.index(f);

    // Get alignment of each local facet with the local cell.
    uint alignment0 = cell0.alignment(f.dim(), facet0);
    uint alignment1 = cell1.alignment(f.dim(), facet1);

    // In principle there are four combinations of alignments:
    // a0 = 0 a1 = 0 ; a0 = 1 a1 = 0 ; a0 = 1 a1 = 0 ; a0 = 1 a1 = 1
    // But only two are not the same.
    // Return a unique alignment for the facet with respect to both cells.
    return pair_alignments[alignment0][alignment1];
  }
  else
  {
    // Mapping from pair of global alignments to alignment of facet pair
    static unsigned int pair_alignments[6][6] = {{0, 1, 4, 3, 2, 5},
                                                {1, 0, 5, 2, 3, 4},
                                                {2, 5, 0, 1, 4, 3},
                                                {3, 4, 1, 0, 5, 2},
                                                {4, 3, 2, 5, 0, 1},
                                                {5, 2, 3, 4, 1, 0}};


    // Get local index of facet with respect to each cell.
    uint facet0 = cell0.index(f);
    uint facet1 = cell1.index(f);

    // Get alignment of each local facet with the local cell.
    uint alignment0 = cell0.alignment(f.dim(), facet0);
    uint alignment1 = cell1.alignment(f.dim(), facet1);

    // Get alignment of facet pair from global alignments
    return pair_alignments[alignment0][alignment1];
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Temporary for benchmarking against old assembly without DofMap
//-----------------------------------------------------------------------------
void FEM::assembleOld(BilinearForm& a, GenericMatrix& A, Mesh& mesh)
{
  assembleCommonOld(&a, 0, 0, &A, 0, 0, mesh);
}
//-----------------------------------------------------------------------------
void FEM::assembleCommonOld(BilinearForm* a, LinearForm* L, Functional* M,
                            GenericMatrix* A, GenericVector* b, real* val,
                            Mesh& mesh)
{
  // Check that the mesh matches the forms
  if ( a ) checkDimensions(*a, mesh);
  if ( L ) checkDimensions(*L, mesh);
  // FIXME: Add dimension check for M

  // Initialize connectivity
  initConnectivity(mesh);

  // Create affine map
  AffineMap map;
  
  // Initialize global data
  uint nz = 0;
  bool interior_contribution = false;
  bool boundary_contribution = false;
  if ( a )
  { 
    const uint M = size(mesh, a->test());
    const uint N = size(mesh, a->trial());
    nz = estimateNonZerosOld(mesh, a->trial());
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
      if ( a && a->interior_contribution() )
        assembleElementOld(*a, *A, mesh, *cell, map, -1);
      
      // Assemble linear form
      //if ( L && L->interior_contribution() )
      //  assembleElement(*L, *b, mesh, *cell, map, -1);              
      
      // Assemble functional
      //if ( M && M->interior_contribution() )
      //  assembleElement(*M, *val, map, -1);              
      
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
    Progress p_boundary("Assembling boundary contributions", boundary.numCells());
    for (CellIterator boundary_cell(boundary); !boundary_cell.end(); ++boundary_cell)
    {
      // Create mesh facet corresponding to boundary cell
      Facet mesh_facet(mesh, cell_map(*boundary_cell));
      dolfin_assert(mesh_facet.numEntities(mesh.topology().dim()) == 1);

      // Get cell to which facet belongs (pick first, there is only one)
      Cell mesh_cell(mesh, mesh_facet.entities(mesh.topology().dim())[0]);

      // Get local index of facet with respect to the cell
      uint local_facet_index = mesh_cell.index(mesh_facet);
      
      // Update affine map for facet 
      map.update(mesh_cell);
      
      // Assemble bilinear form
      if ( a && a->boundary_contribution() )
        assembleElementOld(*a, *A, mesh, mesh_cell, map, local_facet_index);              
      
      // Assemble linear form
      //if ( L && L->boundary_contribution() )
      //  assembleElement(*L, *b, mesh, mesh_cell, map, local_facet_index);              
      
      // Assemble functional
      //if ( M && M->boundary_contribution() )
      //  assembleElement(*M, *val, map, local_facet_index);              
      
      // Update progress  
      p_boundary++;
    }
  }
  
  // Complete assembly
  if ( a )
  {
    A->apply();
    countNonZerosOld(*A, nz);
  }
  if ( L )
    b->apply();
}
//-----------------------------------------------------------------------------
void FEM::assembleElementOld(BilinearForm& a, GenericMatrix& A, 
                             const Mesh& mesh, Cell& cell, AffineMap& map,
                             const int facetID)
{
  // Update form
  a.update(cell, map);
  
  // Compute maps from local to global degrees of freedom
  a.test().nodemap(a.test_nodes, cell, mesh);
  a.trial().nodemap(a.trial_nodes, cell, mesh);
  
  // Compute element matrix 
  if ( facetID < 0 )
    a.eval(a.block, map, map.det);
  else
    a.eval(a.block, map, map.det, facetID);
  
  // Add element matrix to global matrix
  A.add(a.block, a.test_nodes, a.test().spacedim(), a.trial_nodes, a.trial().spacedim());
}
//-----------------------------------------------------------------------------
dolfin::uint FEM::estimateNonZerosOld(Mesh& mesh,
                                      const FiniteElement& element)
{
  uint nzmax = 0;

  mesh.init(0, 0);
  
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    nzmax = std::max(nzmax, vertex->numEntities(0)*element.spacedim());
  }

  return nzmax;
}
//-----------------------------------------------------------------------------
void FEM::countNonZerosOld(const GenericMatrix& A, uint nz)
{
  uint nz_actual = A.nzmax();
  if ( nz_actual > nz )
    dolfin_warning("Actual number of nonzero entries exceeds estimated number of nonzero entries.");
  else
    dolfin_info("Maximum number of nonzeros in each row is %d (estimated %d).",
		nz_actual, nz);
}
//-----------------------------------------------------------------------------
void FEM::assembleSimple(BilinearForm& a,
                         std::vector<std::map<int, real> >& A,
                         Mesh& mesh)
{
  // Check that the mesh matches the form
  checkDimensions(a, mesh);

  // Initialize connectivity
  initConnectivity(mesh);

  // Create affine map
  AffineMap map;
  
  // Initialize global data
  const uint M = size(mesh, a.test());
  const uint N = size(mesh, a.trial());
  A.resize(M);
  
  // Initialize A with zeros
  for (uint i = 0; i < M; i++)
  {
    for (std::map<int, real>::iterator it = A[i].begin(); it != A[i].end(); it++)
    {
      it->second = 0.0;
    }
  }

  dolfin_info("Assembling matrix of size %d x %d.", M, N);
  
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update affine map
    map.update(*cell);
    
    // Update form
    a.update(*cell, map);
    
    // Compute maps from local to global degrees of freedom
    a.test().nodemap(a.test_nodes, *cell, mesh);
    a.trial().nodemap(a.trial_nodes, *cell, mesh);
    
    // Compute element matrix
    a.eval(a.block, map, map.det);
  
    // Add element matrix to global matrix
    uint pos = 0;
    for (uint i = 0; i < a.test().spacedim(); i++)
    {
      std::map<int, real>& row = A[a.test_nodes[i]];
      for (uint j = 0; j < a.trial().spacedim(); j++)
      {
        const uint J = a.trial_nodes[j];
        const std::map<int, real>::iterator it = row.find(J);
        if ( it == row.end() )
          row.insert(it, std::map<int, real>::value_type(J, a.block[pos++]));
        else
          it->second += a.block[pos++];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void FEM::assembleSimple(LinearForm& L,
                         std::vector<real>& b,
                         Mesh& mesh)
{
  // Check that the mesh matches the form
  checkDimensions(L, mesh);

  // Initialize connectivity
  initConnectivity(mesh);

  // Create affine map
  AffineMap map;
  
  // Initialize global data
  const uint M = size(mesh, L.test());
  b.resize(M);
  
  // Initialize b with zeros
  for (uint i = 0; i < M; i++)
    b[i] = 0.0;

  dolfin_info("Assembling vector of size %d.", M);
  
  // Iterate over all cells in the mesh
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update affine map
    map.update(*cell);
    
    // Update form
    L.update(*cell, map);
    
    // Compute map from local to global degrees of freedom
    L.test().nodemap(L.test_nodes, *cell, mesh);
    
    // Compute element vector
    L.eval(L.block, map, map.det);
  
    // Add element vector to global vector
    for (uint i = 0; i < L.test().spacedim(); i++)
      b[L.test_nodes[i]] += L.block[i];
  }
}
//-----------------------------------------------------------------------------
