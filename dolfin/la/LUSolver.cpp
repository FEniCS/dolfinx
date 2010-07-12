// Copyright (C) 20010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-07-11
// Last changed:

#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/common/Timer.h>
#include "CholmodCholeskySolver.h"
#include "UmfpackLUSolver.h"
#include "PETScLUSolver.h"
#include "EpetraLUSolver.h"
#include "LUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LUSolver::LUSolver(std::string type)
{
  // Set default parameters
  parameters = default_parameters();

  // Get linear algebra backend
  const std::string backend = dolfin::parameters["linear_algebra_backend"];

  // Create suitable LU solver
  if (backend == "uBLAS" || backend == "MTL4")
    solver.reset(new UmfpackLUSolver());
  else if (backend == "PETSc")
    #ifdef HAS_PETSC
    solver.reset(new PETScLUSolver());
    #else
    error("PETSc not installed.");
    #endif
  else if (backend == "Epetra")
    error("EpetraLUSolver needs to be updated.");
    //solver.reset(new EpetraLUSolver());
  else
    error("No suitable LU solver for linear algebra backend.");

  solver->parameters.update(parameters);
}
//-----------------------------------------------------------------------------
LUSolver::LUSolver(const GenericMatrix& A, std::string type)
{
  // Set default parameters
  parameters = default_parameters();

  // Get linear algebra backend
  const std::string backend = dolfin::parameters["linear_algebra_backend"];

  // Create suitable LU solver
  if (backend == "uBLAS" || backend == "MTL4")
    solver.reset(new UmfpackLUSolver(A));
  else if (backend == "PETSc")
    #ifdef HAS_PETSC
    solver.reset(new PETScLUSolver(A));
    #else
    error("PETSc not installed.");
    #endif
  else if (backend == "Epetra")
    error("EpetraLUSolver needs to be updated.");
    //solver.reset(new EpetraLUSolver());
  else
    error("No suitable LU solver for linear algebra backend.");

  solver->parameters.update(parameters);
}
//-----------------------------------------------------------------------------
LUSolver::~LUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void LUSolver::set_operator(const GenericMatrix& A)
{
  assert(solver);
  solver->set_operator(A);
}
//-----------------------------------------------------------------------------
dolfin::uint LUSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(solver);
  return solver->solve(x, b);
}
//-----------------------------------------------------------------------------
void LUSolver::factorize()
{
  assert(solver);
  solver->factorize();
}
//-----------------------------------------------------------------------------
dolfin::uint LUSolver::solve_factorized(GenericVector& x,
                                        const GenericVector& b) const
{
  assert(solver);
  return solver->solve_factorized(x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint LUSolver::solve(const GenericMatrix& A, GenericVector& x,
                             const GenericVector& b)
{
  assert(solver);

  Timer timer("LU solver");
  return solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------





