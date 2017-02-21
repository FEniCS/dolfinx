// Copyright (C) 2005-2008 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2005-2009
// Modified by Martin Alnes 2008
// Modified by Johan Hake 2010
//
// First added:  2005-10-23
// Last changed: 2014-05-27

#include <cmath>
#include <string>

#include <dolfin/common/constants.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/GenericLinearSolver.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include "NonlinearProblem.h"
#include "NewtonSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters NewtonSolver::default_parameters()
{
  Parameters p("newton_solver");

  p.add("linear_solver",           "default");
  p.add("preconditioner",          "default");
  p.add("maximum_iterations",      50);
  p.add("relative_tolerance",      1e-9);
  p.add("absolute_tolerance",      1e-10);
  p.add("convergence_criterion",   "residual");
  p.add("method",                  "full");
  p.add("relaxation_parameter",    1.0);
  p.add("report",                  true);
  p.add("error_on_nonconvergence", true);

  //p.add("reuse_preconditioner", false);

  p.add(LUSolver::default_parameters());
  p.add(KrylovSolver::default_parameters());

  return p;
}
//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(MPI_Comm comm)
  : Variable("Newton solver", "unnamed"), _newton_iteration(0), _residual(0.0),
    _residual0(0.0), _solver(nullptr), _matA(std::make_shared<Matrix>(comm)),
    _matP(std::make_shared<Matrix>(comm)), _dx(std::make_shared<Vector>(comm)),
    _b(std::make_shared<Vector>(comm)), _mpi_comm(comm)
{
  // Set default parameters
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(MPI_Comm comm,
                           std::shared_ptr<GenericLinearSolver> solver,
                           GenericLinearAlgebraFactory& factory)
  : Variable("Newton solver", "unnamed"), _newton_iteration(0), _residual(0.0),
    _residual0(0.0), _solver(solver), _matA(factory.create_matrix(comm)),
    _matP(factory.create_matrix(comm)), _dx(factory.create_vector(comm)),
    _b(factory.create_vector(comm)), _mpi_comm(comm)
{
  // Set default parameters
  parameters = default_parameters();

  // Store linear solver type
  parameters["linear_solver"] = "user_defined";
  parameters["preconditioner"] = "user_defined";
}
//-----------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, bool>
NewtonSolver::solve(NonlinearProblem& nonlinear_problem,
                    GenericVector& x)
{
  dolfin_assert(_matA);
  dolfin_assert(_b);
  dolfin_assert(_dx);

  // Extract parameters
  const std::string convergence_criterion = parameters["convergence_criterion"];
  const std::size_t maxiter = parameters["maximum_iterations"];

  // Create linear solver if not already created
  const std::string solver_type = parameters["linear_solver"];
  const std::string pc_type = parameters["preconditioner"];
  if (!_solver)
  {
    _solver = std::make_shared<LinearSolver>(x.mpi_comm(), solver_type, pc_type);
  }
  dolfin_assert(_solver);

  // Set parameters for linear solver
  _solver->update_parameters(parameters(_solver->parameter_type()));

  // Reset iteration counts
  std::size_t krylov_iterations = 0;
  _newton_iteration = 0;

  // Compute F(u)
  nonlinear_problem.form(*_matA, *_matP, *_b, x);
  nonlinear_problem.F(*_b, x);

  // Check convergence
  bool newton_converged = false;
  if (convergence_criterion == "residual")
    newton_converged = converged(*_b, nonlinear_problem, 0);
  else if (convergence_criterion == "incremental")
  {
    // We need to do at least one Newton step with the ||dx||-stopping
    // criterion.
    newton_converged = false;
  }
  else
  {
    dolfin_error("NewtonSolver.cpp",
                 "check for convergence",
                 "The convergence criterion %s is unknown, known criteria are 'residual' or 'incremental'",
                 convergence_criterion.c_str());
  }

  // Get relaxation parameter
  const double relaxation = parameters["relaxation_parameter"];

  // Start iterations
  while (!newton_converged && _newton_iteration < maxiter)
  {
    // Compute Jacobian
    nonlinear_problem.J(*_matA, x);
    nonlinear_problem.J_pc(*_matP, x);

    // Setup (linear) solver (including set operators)
    solver_setup(_matA, _matP, nonlinear_problem, _newton_iteration);

    // Perform linear solve and update total number of Krylov
    // iterations
    if (!_dx->empty())
      _dx->zero();
    krylov_iterations += _solver->solve(*_dx, *_b);

    // Update solution
    if (std::abs(1.0 - relaxation) < DOLFIN_EPS)
      x -= (*_dx);
    else
      x.axpy(-relaxation, *_dx);

    // Increment iteration count
    _newton_iteration++;

    // FIXME: This step is not needed if residual is based on dx and
    //        this has converged.
    // FIXME: But, this function call may update internal variable, etc.
    // Compute F
    nonlinear_problem.form(*_matA, *_matP, *_b, x);
    nonlinear_problem.F(*_b, x);

    // Test for convergence
    if (convergence_criterion == "residual")
      newton_converged = converged(*_b, nonlinear_problem, _newton_iteration);
    else if (convergence_criterion == "incremental")
    {
      // Subtract 1 to make sure that the initial residual0 is
      // properly set.
      newton_converged = converged(*_dx, nonlinear_problem,
                                   _newton_iteration - 1);
    }
    else
    {
      dolfin_error("NewtonSolver.cpp",
                   "check for convergence",
                   "The convergence criterion %s is unknown, known criteria are 'residual' or 'incremental'",
                   convergence_criterion.c_str());
    }
  }

  if (newton_converged)
  {
    if (dolfin::MPI::rank(_mpi_comm) == 0)
    {
      info("Newton solver finished in %d iterations and %d linear solver iterations.",
           _newton_iteration, krylov_iterations);
    }
  }
  else
  {
    const bool error_on_nonconvergence = parameters["error_on_nonconvergence"];
    if (error_on_nonconvergence)
    {
      if (_newton_iteration == maxiter)
      {
        dolfin_error("NewtonSolver.cpp",
                     "solve nonlinear system with NewtonSolver",
                     "Newton solver did not converge because maximum number of iterations reached");
      }
      else
      {
        dolfin_error("NewtonSolver.cpp",
                     "solve nonlinear system with NewtonSolver",
                     "Newton solver did not converge");
      }
    }
    else
      warning("Newton solver did not converge.");
  }

  return std::make_pair(_newton_iteration, newton_converged);
}
//-----------------------------------------------------------------------------
std::size_t NewtonSolver::iteration() const
{
  return _newton_iteration;
}
//-----------------------------------------------------------------------------
double NewtonSolver::residual() const
{
  return _residual;
}
//-----------------------------------------------------------------------------
double NewtonSolver::relative_residual() const
{
  return _residual/_residual0;
}
//-----------------------------------------------------------------------------
GenericLinearSolver& NewtonSolver::linear_solver() const
{
  if (!_solver)
  {
    dolfin_error("NewtonSolver.cpp",
                 "access linear solver for Newton solver",
                 "The linear solver will not be initialized until solve() "
                 "has been called. For control of the linear solver, pass "
                 "a linear solver to the constructor of NewtonSolver");
  }

  return *_solver;
}
//-----------------------------------------------------------------------------
bool NewtonSolver::converged(const GenericVector& r,
                             const NonlinearProblem& nonlinear_problem,
                             std::size_t newton_iteration)
{
  const double rtol = parameters["relative_tolerance"];
  const double atol = parameters["absolute_tolerance"];
  const bool report = parameters["report"];

  _residual = r.norm("l2");

  // If this is the first iteration step, set initial residual
  if (newton_iteration == 0)
    _residual0 = _residual;

  // Relative residual
  const double relative_residual = _residual/_residual0;

  // Output iteration number and residual
  if (report && dolfin::MPI::rank(_mpi_comm) == 0)
  {
    info("Newton iteration %d: r (abs) = %.3e (tol = %.3e) r (rel) = %.3e (tol = %.3e)",
         newton_iteration, _residual, atol, relative_residual, rtol);
  }

  // Return true if convergence criterion is met
  if (relative_residual < rtol || _residual < atol)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
void NewtonSolver::solver_setup(std::shared_ptr<const GenericMatrix> A,
                                std::shared_ptr<const GenericMatrix> P,
                                const NonlinearProblem& nonlinear_problem,
                                std::size_t interation)
{
  // Update Jacobian in linear solver (and preconditioner if given)
  if (_matP->empty())
    _solver->set_operator(A);
  else
    _solver->set_operators(A, P);
}
//-----------------------------------------------------------------------------
