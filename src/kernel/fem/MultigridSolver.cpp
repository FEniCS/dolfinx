// Copyright (C) 2004 Andreas Mark and Andreas Nilsson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Matrix.h>
#include <dolfin/Solver.h>
#include <dolfin/Vector.h>
#include <dolfin/dolfin_mesh.h>
#include <dolfin/MeshHierarchy.h>
#include <dolfin/MultigridSolver.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/Variable.h>
#include <dolfin/General.h>

using namespace dolfin;

unsigned int noPreSmooth  = dolfin_get("multigrid pre-smoothing");
unsigned int noPostSmooth = dolfin_get("multigrid post-smoothing");

//-----------------------------------------------------------------------------
void MultigridSolver::solve(PDE& pde, Vector& x, MeshHierarchy& meshes) 
{
  if (meshes.size() <= 1)
    dolfin_error("Found no mesh hierarchy.");
  
  dolfin_info("Found %d levels in the mesh", meshes.size());
  
  mainIteration(pde, x, meshes);
}
//-----------------------------------------------------------------------------
void MultigridSolver::solve(PDE& pde, Vector& x, Mesh& mesh) 
{
  MeshHierarchy meshes(mesh);
  
  if (meshes.size() <= 1)
    dolfin_error("Found no mesh hierarchy.");
  
  dolfin_info("Found %d levels in the mesh",meshes.size());
  
  mainIteration(pde, x, meshes);
}
//-----------------------------------------------------------------------------
void MultigridSolver::solve(PDE& pde, Vector& x, Mesh& mesh, 
			    unsigned int refinements)
{ 
  // Make some uniform refinements
  for (unsigned int i = 0; i < refinements; i++)
    mesh.refineUniformly();  
  
  // Create mesh hierarchy
  MeshHierarchy meshes(mesh);
  
  mainIteration(pde, x, meshes);
}
//-----------------------------------------------------------------------------
void MultigridSolver::mainIteration(PDE& pde, Vector& x, 
				    const MeshHierarchy& meshes)
{
  // Here is the work common for all different versions of solve().
  
  unsigned int noLevels = meshes.size();
  unsigned int max_it = dolfin_get("multigrid iterations");
  real tol = dolfin_get("multigrid tolerance");
  
  Matrices A(noLevels); // Matrices, and vectors for the A's and b's
  Vectors  b(noLevels);  // for different meshes. 
  
  // Assemble the matrices
  dolfin_info("Assembling for multigrid");
  for (unsigned int i=0; i < noLevels ; i++)
    FEM::assemble(pde, meshes(i), A[i], b[i]);   
 
  tic();
  
  x.init(b[noLevels-1].size()); // initial guess x = 0
  Vector R(max_it);
  Vector rfine(b[noLevels-1].size());
  
  A[noLevels-1].mult(x,rfine); // computes residual on finest mesh using
  rfine-=b[noLevels-1];        // the initial guess x.
 
  int unsigned i = 0;
  for (; i < max_it; i++)
  {   
    fullVCycle(x, rfine, meshes, A);
    
    A[noLevels-1].mult(x,rfine);
    rfine-=b[noLevels-1];  	
    R(i) = rfine.norm(2);
    
    dolfin_info("Multigrid V-cycle iteration %d: residual = %e", i, R(i));
    if (R(i) < tol)
      break;
  }
  
  dolfin_info("Multigrid converged in %d iterations.", i + 1);

  tocd();
}
//-----------------------------------------------------------------------------
void MultigridSolver::fullVCycle(Vector& x, const Vector& r_fine,
				 const MeshHierarchy& meshes,
				 const Matrices& A)
{	      
  unsigned int noLevels = meshes.size();
  Vector z;            // correction to the initial guess x
  Vectors r(noLevels); // residuals on all meshes
  Vector tmp; 
  r[noLevels-1] = r_fine;
   
  for (int level = noLevels - 2; level >=  0; level--) {
    tmp = r[level+1];  // we need tmp not to destroy r[level+1]
    restrict(tmp, meshes, level+1);
    r[level] = tmp;
  }
  
  Matrix A0(Matrix::dense); // We want a dense matrix so that 
  A0 = A[0];                // we can get the exact solution
  A0.solve(z, r[0]);  
  
  // Compute correction
  for (unsigned int level = 1; level < noLevels ; level++)
  {
    interpolate(z, meshes, level-1);
    vCycle(z, r[level], meshes, A, level);
  }
  
  // Add correction to initial guess
  x -= z;
}
//----------------------------------------------------------------------------
void MultigridSolver::vCycle(Vector& x, const Vector& rhs,
			     const MeshHierarchy& meshes, const Matrices& A,
			     unsigned int level)
{
  if ( level == 0 )
  {
    Matrix A0(Matrix::dense);
    A0 = A[0];
    A0.solve(x, rhs); 
  }
  else
  {
    Vector r, zeros(meshes(level-1).noNodes()); 
    smooth(x, rhs, noPreSmooth, A, level); 
    
    // Compute Residual
    A[level].mult(x,r);
    r-=rhs;  
    
    restrict(r, meshes, level);
    vCycle(r, zeros, meshes, A, level-1);
    interpolate(r , meshes, level-1 );
    x-=r; 

    smooth(x, rhs, noPostSmooth, A, level);    
  }
}
//---------------------------------------------------------------------------
void MultigridSolver::smooth(Vector& x, const Vector& rhs, 
			     unsigned int noSmoothings, const Matrices& A,
			     unsigned int level)
{
  real aii = 0.0;
  real aij = 0.0;
  unsigned int j;	
  
  for (unsigned int m = 0; m < noSmoothings ; m++)
  {
    // Gauss-Seidel, code from SISolver
    for (unsigned int i = 0; i < A[level].size(0); i++) {
      x(i) = rhs(i);
      for (unsigned int pos = 0; !A[level].endrow(i,pos); pos++) {
	aij = A[level](i,j,pos);
	if (i==j) aii = aij;
	else x(i) -= aij*x(j);
      }
      x(i) /= aii;
    }
  }	
}
//----------------------------------------------------------------------------
void MultigridSolver::restrict(Vector& xf, const MeshHierarchy& meshes,
                               unsigned int level)
{
  // Restrict from i -> i-1
  
  Vector xc(meshes(level-1).noNodes());
  
  unsigned int noNeighbors;
  double tmp=0;
  Node* childnode;

  // Iterate over nodes in coarse mesh
  for (NodeIterator cnode(meshes(level-1)); !cnode.end() ; ++cnode)
  { 
    childnode = cnode->child();
    noNeighbors = childnode->noNodeNeighbors();
    
    // Iterate over neighbors in finemesh
    for (NodeIterator neighbor(*childnode); !neighbor.end() ; ++neighbor)
    { 
      if (neighbor == childnode)
	tmp += 2*xf(neighbor->id());
      else
	tmp += xf(neighbor->id()); 
    }
    xc(cnode->id()) = tmp/(noNeighbors + 1);
    tmp=0;
  }
  
  // We want to "return" xc 
  xf = xc;
}
//----------------------------------------------------------------------------
void MultigridSolver::interpolate(Vector& xc, const MeshHierarchy& meshes, 
			          unsigned int level)
{
  unsigned int k=0;
  real tmp=0;
  Node* cnode; // node on the course grid
  Vector xf(meshes(level+1).noNodes());   // the new x on the fine grid

  // Iterate in fine mesh
  for (NodeIterator fnode(meshes(level+1)); !fnode.end() ; ++fnode)
  {
    cnode = fnode->parent();       
	  
    if (cnode)
        xf(fnode->id()) = xc(cnode->id());
    else
    {
      for (NodeIterator neighbor(fnode); !neighbor.end(); ++neighbor)
      { 
	cnode = neighbor->parent();	   
	if (cnode)
	{
	  tmp += xc(cnode->id());   
	  k++;
	}		
      }
      xf(fnode->id()) = tmp/k; 
      k=0;
      tmp=0;
    }
  }
  
  xc = xf;
}
//------------------------------------------------------------------------------
