// Copyright (C) 2008-2011 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Marie E. Rognes, 2011.
//
// First added:  2011-01-14 (2008-12-26 as VariationalProblem.cpp)
// Last changed: 2011-01-14

#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/function/Function.h>
#include "assemble.h"
#include "VariationalProblem.h"
#include "NonlinearVariationalSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NonlinearVariationalSolver::NonlinearVariationalSolver()
{
  // Set default parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
NonlinearVariationalSolver::~NonlinearVariationalSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::solve(const VariationalProblem& problem,
                                       Function& u)
{
  begin("Solving nonlinear variational problem.");

  // Create nonlinear problem
  _NonlinearProblem nonlinear_problem(problem, parameters);

  // Create Newton solver annond set parameters
  NewtonSolver newton_solver;
  newton_solver.parameters.update(parameters("newton_solver"));

  // Solve nonlinear problem using Newton's method
  newton_solver.solve(nonlinear_problem, u.vector());

  end();
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::solve(const VariationalProblem& problem,
                                       Function& u0,
                                       Function& u1)
{
  // Create function
  Function u(problem.trial_space());

  // Solve variational problem
  solve(problem, u);

  // Extract subfunctions
  u0 = u[0];
  u1 = u[1];
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::solve(const VariationalProblem& problem,
                                       Function& u0,
                                       Function& u1,
                                       Function& u2)
{
  // Create function
  Function u(problem.trial_space());

  // Solve variational problem
  solve(problem, u);

  // Extract subfunctions
  u0 = u[0];
  u1 = u[1];
  u2 = u[2];
}
//-----------------------------------------------------------------------------
// Implementation of _NonlinearProblem
//-----------------------------------------------------------------------------
NonlinearVariationalSolver::
_NonlinearProblem::_NonlinearProblem(const VariationalProblem& problem,
                                     const Parameters& parameters)
  : problem(problem), parameters(parameters), jacobian_initialized(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonlinearVariationalSolver::
_NonlinearProblem::~_NonlinearProblem()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::
_NonlinearProblem::F(GenericVector& b, const GenericVector& x)
{
  // Assemble right-hand side
  assemble(b, problem.linear_form(),
           problem.cell_domains(),
           problem.exterior_facet_domains(),
           problem.interior_facet_domains());

  // Apply boundary conditions
  for (uint i = 0; i < problem.bcs().size(); i++)
    problem.bcs()[i]->apply(b, x);

  // Print vector
  const bool print_rhs = parameters["print_rhs"];
  if (print_rhs == true)
    info(b, true);
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::
_NonlinearProblem::J(GenericMatrix& A, const GenericVector& x)
{
  // Check if Jacobian matrix sparsity pattern should be reset
  bool reset_sparsity = !(parameters["reset_jacobian"] && jacobian_initialized);

  // Assemble left-hand side
  assemble(A, problem.bilinear_form(),
           problem.cell_domains(),
           problem.exterior_facet_domains(),
           problem.interior_facet_domains(),
           reset_sparsity);

  // Remember that Jacobian has been initialized
  jacobian_initialized = true;

  // Apply boundary conditions
  for (uint i = 0; i < problem.bcs().size(); i++)
    problem.bcs()[i]->apply(A);

  // Print matrix
  const bool print_matrix = parameters["print_matrix"];
  if (print_matrix == true)
    info(A, true);
}
//-----------------------------------------------------------------------------
