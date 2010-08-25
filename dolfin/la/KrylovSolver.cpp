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
  p.add("report",                  true); /* deprecate? */
  p.add("monitor_convergence",     false);
  p.add("error_on_nonconvergence", true);
  p.add("nonzero_initial_guess", false);

  // GMRES options
  Parameters p_gmres("gmres");
  p_gmres.add("restart", 30);

  // ILU preconditioner options
  Parameters p_pc_ilu("ilu");
  p_pc_ilu.add("fill_level", 0);

  // Schwartz preconditioner options
  Parameters p_pc_schartz("schwarz");
  p_pc_schartz.add("overlap", 1);

  // Preconditioner options
  Parameters p_pc("preconditioner");
  p_pc.add("shift_nonzero",        0.0);
  p_pc.add("reuse",                false);
  p_pc.add("same_nonzero_pattern", false);
  p_pc.add("report",               false);
  p_pc.add(p_pc_ilu);


  // Add nested parameters
  p.add(p_gmres);
  p.add(p_pc);

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
void KrylovSolver::set_operators(const GenericMatrix& A, const GenericMatrix& P)
{
  assert(solver);
  solver->parameters.update(parameters);
  solver->set_operators(A, P);
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
