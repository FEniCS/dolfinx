// Copyright (C) 2007-2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2007-07-03
// Last changed: 2010-07-18

#include <dolfin/common/Timer.h>
#include <dolfin/parameter/Parameters.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "DefaultFactory.h"
#include "KrylovSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters KrylovSolver::default_parameters()
{
  Parameters p("krylov_solver");

  p.add("relative_tolerance",      1e-6);
  p.add("absolute_tolerance",      1e-15);
  p.add("divergence_limit",        1e4);
  p.add("maximum_iterations",      10000);
  p.add("gmres_restart",           30);
  p.add("shift_nonzero",           0.0);
  p.add("report",                  true); /* deprecate? */
  p.add("monitor_convergence",     false);
  p.add("error_on_nonconvergence", true);

  p.add("reuse_preconditioner", false);
  p.add("same_nonzero_pattern", false);
  p.add("nonzero_initial_guess", false);

  return p;
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(std::string solver_type, std::string pc_type)
{
  // Set default parameters
  parameters = default_parameters();

  DefaultFactory factory;
  solver.reset( factory.create_krylov_solver(solver_type, pc_type) );
  solver->parameters.update(parameters);
}
//-----------------------------------------------------------------------------
KrylovSolver::~KrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void KrylovSolver::set_operator(const GenericMatrix& A)
{
  assert(solver);
  solver->parameters.update(parameters);
  solver->set_operator(A);
}
//-----------------------------------------------------------------------------
dolfin::uint KrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(solver);
  Timer timer("Krylov solver");
  solver->parameters.update(parameters);
  return solver->solve(x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint KrylovSolver::solve(const GenericMatrix& A, GenericVector& x,
                                 const GenericVector& b)
{
  assert(solver);
  Timer timer("Krylov solver");
  solver->parameters.update(parameters);
  return solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------
