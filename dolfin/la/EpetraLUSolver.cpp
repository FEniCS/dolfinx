// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2008-05-16

#ifdef HAS_TRILINOS

#include "GenericMatrix.h"
#include "GenericVector.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "EpetraLUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraLUSolver::EpetraLUSolver() 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraLUSolver::~EpetraLUSolver() 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraLUSolver::solve(const GenericMatrix& A, GenericVector& x, 
                                       const GenericVector& b) 
{
  return  solve(A.down_cast<EpetraMatrix>(), x.down_cast<EpetraVector>(), 
                b.down_cast<EpetraVector>());
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraLUSolver::solve(const EpetraMatrix&A, EpetraVector& x, 
                                   const EpetraVector& b)
{
  error("EpetraLUSolver::solve not implemented"); 
  return 0; 
}
//-----------------------------------------------------------------------------
void EpetraLUSolver::disp() const 
{
  error("EpetraLUSolver::disp not implemented"); 
}
//-----------------------------------------------------------------------------
#endif 

