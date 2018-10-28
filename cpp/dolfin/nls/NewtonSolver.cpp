// Copyright (C) 2005-2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "NewtonSolver.h"
#include "NonlinearProblem.h"
#include <cmath>
#include <dolfin/common/MPI.h>
#include <dolfin/common/constants.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScOptions.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/log/log.h>
#include <string>

using namespace dolfin;

//-----------------------------------------------------------------------------
parameter::Parameters dolfin::nls::NewtonSolver::default_parameters()
{
  parameter::Parameters p("newton_solver");

  p.add("maximum_iterations", 50);
  p.add("relative_tolerance", 1e-9);
  p.add("absolute_tolerance", 1e-10);
  p.add("convergence_criterion", "residual");
  p.add("report", true);
  p.add("error_on_nonconvergence", true);
  p.add<double>("relaxation_parameter");

  return p;
}
//-----------------------------------------------------------------------------
dolfin::nls::NewtonSolver::NewtonSolver(MPI_Comm comm)
    : common::Variable("Newton solver"), _krylov_iterations(0),
      _relaxation_parameter(1.0), _residual(0.0), _residual0(0.0),
      _mpi_comm(comm)
{
  // Set default parameters
  parameters = default_parameters();

  // Create linear solver if not already created. Default to LU.
  _solver = std::make_shared<la::PETScKrylovSolver>(comm);
  _solver->set_options_prefix("nls_solve_");
  la::PETScOptions::set("nls_solve_ksp_type", "preonly");
  la::PETScOptions::set("nls_solve_pc_type", "lu");
#if PETSC_HAVE_MUMPS
  la::PETScOptions::set("nls_solve_pc_factor_mat_solver_type", "mumps");
#endif
  _solver->set_from_options();
}
//-----------------------------------------------------------------------------
dolfin::nls::NewtonSolver::~NewtonSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::pair<int, bool>
dolfin::nls::NewtonSolver::solve(NonlinearProblem& nonlinear_problem,
                                 la::PETScVector& x)
{
  // Extract parameters
  const std::string convergence_criterion = parameters["convergence_criterion"];
  const int maxiter = parameters["maximum_iterations"];
  if (parameters["relaxation_parameter"].is_set())
    set_relaxation_parameter(parameters["relaxation_parameter"]);

  // Reset iteration counts
  int newton_iteration = 0;
  _krylov_iterations = 0;

  // Compute F(u) (assembled into _b)
  la::PETScMatrix* A = nullptr;
  la::PETScMatrix* P = nullptr;
  la::PETScVector* b = nullptr;

  x.update_ghosts();
  nonlinear_problem.form(x);
  b = nonlinear_problem.F(x);
  assert(b);
  // double b_norm0 = b->norm(la::Norm::l2);
  // std::cout << "b0 norm: " << b_norm0 << std::endl;

  // Check convergence
  bool newton_converged = false;
  if (convergence_criterion == "residual")
    newton_converged = converged(*b, nonlinear_problem, 0);
  else if (convergence_criterion == "incremental")
  {
    // We need to do at least one Newton step with the ||dx||-stopping
    // criterion.
    newton_converged = false;
  }
  else
  {
    log::dolfin_error(
        "NewtonSolver.cpp", "check for convergence",
        "The convergence criterion %s is unknown, known criteria are "
        "'residual' or 'incremental'",
        convergence_criterion.c_str());
  }

  // Start iterations
  while (!newton_converged && newton_iteration < maxiter)
  {
    // Compute Jacobian
    // double x_norm0 = x.norm(la::Norm::l2);
    // std::cout << "x0 norm: " << x_norm0 << std::endl;

    A = nonlinear_problem.J(x);
    assert(A);
    P = nonlinear_problem.P(x);

    // double A_norm = A->norm(la::Norm::frobenius);
    // std::cout << "A norm: " << A_norm << std::endl;

    if (!_dx)
      _dx = std::make_unique<la::PETScVector>(A->init_vector(1));

    // Set operators
    assert(_solver);
    if (P)
      _solver->set_operators(*A, *P);
    else
      _solver->set_operator(*A);

    // Perform linear solve and update total number of Krylov iterations
    //_dx->set(0.0);
    _krylov_iterations += _solver->solve(*_dx, *b);
    //_dx->apply();

    // double dx_norm = _dx->norm(la::Norm::l2);
    // std::cout << "dx norm: " << dx_norm << std::endl;

    // Update solution
    update_solution(x, *_dx, _relaxation_parameter, nonlinear_problem,
                    newton_iteration);
    x.update_ghosts();

    // Increment iteration count
    ++newton_iteration;

    // FIXME: This step is not needed if residual is based on dx and
    //        this has converged.
    // FIXME: But, this function call may update internal variable, etc.
    // Compute F

    // x.update_ghosts();
    // VecView(x.vec(), PETSC_VIEWER_STDOUT_WORLD);
    // double x_norm1 = x.norm(la::Norm::l2);
    // std::cout << "x1 norm: " << x_norm1 << std::endl;
    nonlinear_problem.form(x);
    b = nonlinear_problem.F(x);

    // double b_norm1 = b->norm(la::Norm::l2);
    // std::cout << "b1 norm: " << b_norm1 << std::endl;

    // Test for convergence
    if (convergence_criterion == "residual")
      newton_converged = converged(*b, nonlinear_problem, newton_iteration);
    else if (convergence_criterion == "incremental")
    {
      // Subtract 1 to make sure that the initial residual0 is
      // properly set.
      newton_converged
          = converged(*_dx, nonlinear_problem, newton_iteration - 1);
    }
    else
      throw std::runtime_error("Unknown convergence criterion string.");
  }

  if (newton_converged)
  {
    if (_mpi_comm.rank() == 0)
    {
      log::info("Newton solver finished in %d iterations and %d linear solver "
                "iterations.",
                newton_iteration, _krylov_iterations);
    }
  }
  else
  {
    // const bool error_on_nonconvergence =
    // parameters["error_on_nonconvergence"];
    const bool error_on_nonconvergence = false;
    if (error_on_nonconvergence)
    {
      if (newton_iteration == maxiter)
      {
        throw std::runtime_error("Newton solver did not converge because "
                                 "maximum number of iterations reached");
      }
      else
        throw std::runtime_error("Newton solver did not converge");
    }
    else
      log::warning("Newton solver did not converge.");
  }

  return std::make_pair(newton_iteration, newton_converged);
}
//-----------------------------------------------------------------------------
int dolfin::nls::NewtonSolver::krylov_iterations() const
{
  return _krylov_iterations;
}
//-----------------------------------------------------------------------------
double dolfin::nls::NewtonSolver::residual() const { return _residual; }
//-----------------------------------------------------------------------------
double dolfin::nls::NewtonSolver::residual0() const { return _residual0; }
//-----------------------------------------------------------------------------
double dolfin::nls::NewtonSolver::relative_residual() const
{
  return _residual / _residual0;
}
//-----------------------------------------------------------------------------
bool dolfin::nls::NewtonSolver::converged(
    const la::PETScVector& r, const NonlinearProblem& nonlinear_problem,
    std::size_t newton_iteration)
{
  const double rtol = parameters["relative_tolerance"];
  const double atol = parameters["absolute_tolerance"];
  const bool report = parameters["report"];

  _residual = r.norm(la::Norm::l2);

  // If this is the first iteration step, set initial residual
  if (newton_iteration == 0)
    _residual0 = _residual;

  // Relative residual
  const double relative_residual = _residual / _residual0;

  // Output iteration number and residual
  if (report && _mpi_comm.rank() == 0)
  {
    log::info("Newton iteration %d: r (abs) = %.3e (tol = %.3e) r (rel) = "
              "%.3e (tol "
              "= %.3e)",
              newton_iteration, _residual, atol, relative_residual, rtol);
  }

  // Return true if convergence criterion is met
  if (relative_residual < rtol || _residual < atol)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
void dolfin::nls::NewtonSolver::update_solution(
    la::PETScVector& x, const la::PETScVector& dx, double relaxation_parameter,
    const NonlinearProblem& nonlinear_problem, std::size_t interation)
{
  x.axpy(-relaxation_parameter, dx);
}
//-----------------------------------------------------------------------------
