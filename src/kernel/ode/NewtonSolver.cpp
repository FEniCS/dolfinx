// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Alloc.h>
#include <dolfin/ODE.h>
#include <dolfin/NewTimeSlab.h>
#include <dolfin/NewMethod.h>
#include <dolfin/NewtonSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(ODE& ode, NewTimeSlab& timeslab, const NewMethod& method)
  : TimeSlabSolver(ode, timeslab, method), f(0), A(ode, timeslab, method)
{
  // Initialize local array
  f = new real[method.qsize()];
}
//-----------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
  // Delete local array
  if ( f ) delete [] f;
}
//-----------------------------------------------------------------------------
void NewtonSolver::start()
{
  // Get size of system
  int nj = static_cast<int>(ts.nj);

  // Initialize increment vector
  dx.init(nj);

  // Initialize Jacobian matrix
  A.init(dx, dx);

  // Compute Jacobian
  A.update(ts.starttime());
}
//-----------------------------------------------------------------------------
real NewtonSolver::iteration()
{
  // Evaluate F at current x
  Feval();
  
  // Solve linear system F for dx
  //solver.solve(A, dx, F);

  // Get array containing the increments (assumes uniprocessor case)
  real* dxvals = dx.array();

  // Update solution x -> x - dx
  for (uint j = 0; j < ts.nj; j++)
    ts.jx[j] -= dxvals[j];

  // Restor array
  dx.restore(dxvals);
  
  return 0.01;
}
//-----------------------------------------------------------------------------
void NewtonSolver::Feval()
{
  

}
//-----------------------------------------------------------------------------
