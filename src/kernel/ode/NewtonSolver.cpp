// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/Alloc.h>
#include <dolfin/NewTimeSlab.h>
#include <dolfin/NewMethod.h>
#include <dolfin/NewtonSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(NewTimeSlab& timeslab, const NewMethod& method)
  : TimeSlabSolver(timeslab, method), f(0), A(0), x(0)
{
  // Initialize local array
  f = new real[method.qsize()];

  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
  // Delete local array
  if ( f ) delete [] f;

  // Delete PETSc matrix if necessary
  if ( A ) MatDestroy(A);

  // Delete PETSc vector if necessary
  if ( x ) VecDestroy(x);
}
//-----------------------------------------------------------------------------
void NewtonSolver::start()
{
  // Get size of system
  int nj = static_cast<int>(ts.nj);

  // Create matrix and vector the first time
  if ( !A )
  {
    cout << "Creating matrix for the first time" << endl;

    // Create vector
    VecCreate(PETSC_COMM_WORLD, &x);
    VecSetSizes(x, PETSC_DECIDE, nj);
    VecSetFromOptions(x);
    
    // Extract local partitioning needed to create the matrix
    int m = 0;
    VecGetLocalSize(x, &m);
    cout << "m = " << m << endl;
    
    // Create matrix
    MatCreateShell(PETSC_COMM_WORLD, m, nj, nj, nj, (void*) this, &A);
    return;
  }
  

  
  
  

  // Check if we need to change the size of the matrix
  int m = 0;
  int n = 0;
  MatGetSize(A, &m, &n);
  if ( m != nj || n != nj )
  {
    cout << "Need to change size of matrix: m = " << m << " nj = " << nj << endl;
    MatDestroy(A);
    MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nj, nj, 
		   (void*) this, &A);
  }
}
//-----------------------------------------------------------------------------
real NewtonSolver::iteration()
{
  dolfin_error("Not implemented");
  
  // Reset dof
  uint j = 0;

  // Reset current sub slab
  int s = -1;

  // Reset elast
  ts.elast = -1;

  // Reset maximum increment
  real max_increment = 0.0;

  // Iterate over all elements
  for (uint e = 0; e < ts.ne; e++)
  {
    // Cover all elements in current sub slab
    s = ts.cover(s, e);

    // Get element data
    const uint i = ts.ei[e];
    const real a = ts.sa[s];
    const real b = ts.sb[s];
    const real k = b - a;

    // Get initial value for element
    const int ep = ts.ee[e];
    const uint jp = ep * method.nsize();
    const real x0 = ( ep != -1 ? ts.jx[jp + method.nsize() - 1] : ts.u0[i] );

    // Evaluate right-hand side at quadrature points of element
    ts.feval(f, s, e, i, a, b, k);
    //cout << "f = "; Alloc::disp(f, method.qsize());

    // Update values on element using fixed point iteration
    const real increment = method.update(x0, f, k, ts.jx + j);
    //cout << "x = "; Alloc::disp(ts.jx + j, method.nsize());
    
    // Update maximum increment
    if ( increment > max_increment )
      max_increment = increment;
    
    // Update dof
    j += method.nsize();
  }

  return max_increment;
}
//-----------------------------------------------------------------------------
