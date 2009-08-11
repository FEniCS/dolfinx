// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2009-08-10

#ifdef HAS_TRILINOS

#include "EpetraLUSolver.h"
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "LUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters EpetraLUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("epetra_lu_solver");
  return p;
}
//-----------------------------------------------------------------------------
EpetraLUSolver::EpetraLUSolver()
{
  parameters = default_parameters();
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
std::string EpetraLUSolver::str(bool verbose) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------

#endif

