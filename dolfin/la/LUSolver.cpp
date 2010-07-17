// Copyright (C) 20010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-07-11
// Last changed:

#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/common/Timer.h>
#include "DefaultFactory.h"
#include "LUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LUSolver::LUSolver(std::string type)
{
  // Set default parameters
  parameters = default_parameters();

  DefaultFactory factory;
  solver.reset(factory.create_lu_solver());

  solver->parameters.update(parameters);
}
//-----------------------------------------------------------------------------
LUSolver::LUSolver(const GenericMatrix& A, std::string type)
{
  // Set default parameters
  parameters = default_parameters();

  DefaultFactory factory;
  solver.reset(factory.create_lu_solver());
  solver->set_operator(A);

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
  solver->parameters.update(parameters);
  solver->set_operator(A);
}
//-----------------------------------------------------------------------------
dolfin::uint LUSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(solver);

  Timer timer("LU solver");
  solver->parameters.update(parameters);
  return solver->solve(x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint LUSolver::solve(const GenericMatrix& A, GenericVector& x,
                             const GenericVector& b)
{
  assert(solver);

  Timer timer("LU solver");
  solver->parameters.update(parameters);
  return solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------





