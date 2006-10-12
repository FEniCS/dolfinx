// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Andy Terrel 2005.
// Modified by Garth N. Wells 2005, 2006.
//
// First added:  2004-05-19
// Last changed: 2006-09-18

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
  // Create boundary
  BoundaryMesh boundary(mesh);
  
  assembleCommon(a, L, M, A, b, val, mesh);

}
//-----------------------------------------------------------------------------
void FEM::applyCommonBC(GenericMatrix* A, GenericVector* b, 
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
  switch ( mesh.dim() )
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
  switch ( mesh.dim() )
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
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    nzmax = std::max(nzmax, vertex->numConnections(0)*element.spacedim());

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
