// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Function.h>
#include <dolfin/Vector.h>
#include <dolfin/ODESolver.h>
#include <dolfin/ODE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ODE::ODE(unsigned int N) : sparsity(N)
{
  this->N = N;
  T = 1.0;
}
//-----------------------------------------------------------------------------
ODE::~ODE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
unsigned int ODE::size() const
{
  return N;  
}
//-----------------------------------------------------------------------------
real ODE::endtime() const
{
  return T;
}
//-----------------------------------------------------------------------------
void ODE::solve()
{
  ODESolver::solve(*this);
}
//-----------------------------------------------------------------------------
void ODE::solve(Function& u)
{
  ODESolver::solve(*this, u);
}
//-----------------------------------------------------------------------------
void ODE::solve(Function& u, Function& phi)
{
  ODESolver::solve(*this, u, phi);
}
//-----------------------------------------------------------------------------
void ODE::sparse(unsigned int i, unsigned int size)
{
  sparsity.setsize(i, size);
}
//-----------------------------------------------------------------------------
void ODE::depends(unsigned int i, unsigned int j)
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
