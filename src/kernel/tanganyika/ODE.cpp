// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODESolver.h>
#include <dolfin/ODE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ODE::ODE(int N) : sparsity(N)
{
  this->N = N;
  T = 1.0;

  u0.init(N);
}
//-----------------------------------------------------------------------------
int ODE::size() const
{
  return N;  
}
//-----------------------------------------------------------------------------
void ODE::solve()
{
  ODESolver solver;
  solver.solve(*this, T);
}
//-----------------------------------------------------------------------------
void ODE::sparse(int i, int size)
{
  sparsity.setsize(i, size);
}
//-----------------------------------------------------------------------------
void ODE::depends(int i, int j)
{
  sparsity.set(i,j);
}
//-----------------------------------------------------------------------------
void ODE::sparse(const Matrix& A)
{
  sparsity.set(A);
}
//-----------------------------------------------------------------------------
void ODE::sparse()
{
  sparsity.guess(*this);
}
//-----------------------------------------------------------------------------
