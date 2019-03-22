// Copyright (C) 2005-2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "NewtonSolver.h"
#include "NonlinearProblem.h"
#include <dolfin/common/MPI.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScOptions.h>
#include <dolfin/la/PETScVector.h>
#include <spdlog/spdlog.h>
#include <string>

using namespace dolfin;

//-----------------------------------------------------------------------------
nls::NewtonSolver::NewtonSolver(MPI_Comm comm)
    : _krylov_iterations(0), _residual(0.0), _residual0(0.0), _solver(comm),
      _dx(nullptr), _mpi_comm(comm)
{
  // Create linear solver if not already created. Default to LU.
  _solver.set_options_prefix("nls_solve_");
  la::PETScOptions::set("nls_solve_ksp_type", "preonly");
  la::PETScOptions::set("nls_solve_pc_type", "lu");
#if PETSC_HAVE_MUMPS
  la::PETScOptions::set("nls_solve_pc_factor_mat_solver_type", "mumps");
#endif
  _solver.set_from_options();
}
//-----------------------------------------------------------------------------
nls::NewtonSolver::~NewtonSolver()
{
  if (_dx)
    VecDestroy(&_dx);
}
//-----------------------------------------------------------------------------
std::pair<int, bool>
dolfin::nls::NewtonSolver::solve(NonlinearProblem& nonlinear_problem, Vec x)
{
  // Reset iteration counts
  int newton_iteration = 0;
  _krylov_iterations = 0;

  // Compute F(u) (assembled into _b)
  Mat A(nullptr), P(nullptr);
  Vec b = nullptr;

  nonlinear_problem.form(x);
  b = nonlinear_problem.F(x);
  assert(b);

  // Check convergence
  bool newton_converged = false;
  if (convergence_criterion == "residual")
    newton_converged = converged(b, nonlinear_problem, 0);
  else if (convergence_criterion == "incremental")
  {
    // We need to do at least one Newton step with the ||dx||-stopping
    // criterion.
    newton_converged = false;
  }
  else
  {
    spdlog::error("NewtonSolver.cpp", "check for convergence",
                  "The convergence criterion %s is unknown, known criteria are "
                  "'residual' or 'incremental'",
                  convergence_criterion.c_str());
    throw std::runtime_error("Unknown convergence criterion");
  }

  // Start iterations
  while (!newton_converged and newton_iteration < max_it)
  {
    // Compute Jacobian
    A = nonlinear_problem.J(x);
    assert(A);
    P = nonlinear_problem.P(x);
    if (!P)
      P = A;

    if (!_dx)
      MatCreateVecs(A, &_dx, nullptr);

    // FIXME: check that this is efficient if A and/or P are unchanged
    // Set operators
    _solver.set_operators(A, P);

    // Perform linear solve and update total number of Krylov iterations
    _krylov_iterations += _solver.solve(_dx, b);

    // Update solution
    update_solution(x, _dx, relaxation_parameter, nonlinear_problem,
                    newton_iteration);

    // Increment iteration count
    ++newton_iteration;

    // FIXME: This step is not needed if residual is based on dx and
    //        this has converged.
    // FIXME: But, this function call may update internal variables, etc.
    // Compute F
    nonlinear_problem.form(x);
    b = nonlinear_problem.F(x);

    // Test for convergence
    if (convergence_criterion == "residual")
      newton_converged = converged(b, nonlinear_problem, newton_iteration);
    else if (convergence_criterion == "incremental")
    {
      // Subtract 1 to make sure that the initial residual0 is
      // properly set.
      newton_converged
          = converged(_dx, nonlinear_problem, newton_iteration - 1);
    }
    else
      throw std::runtime_error("Unknown convergence criterion string.");
  }

  if (newton_converged)
  {
    if (_mpi_comm.rank() == 0)
    {
      spdlog::info(
          "Newton solver finished in %d iterations and %d linear solver "
          "iterations.",
          newton_iteration, _krylov_iterations);
    }
  }
  else
  {
    if (error_on_nonconvergence)
    {
      if (newton_iteration == max_it)
      {
        throw std::runtime_error("Newton solver did not converge because "
                                 "maximum number of iterations reached");
      }
      else
        throw std::runtime_error("Newton solver did not converge");
    }
    else
      spdlog::warn("Newton solver did not converge.");
  }

  return std::make_pair(newton_iteration, newton_converged);
}
//-----------------------------------------------------------------------------
int nls::NewtonSolver::krylov_iterations() const { return _krylov_iterations; }
//-----------------------------------------------------------------------------
double nls::NewtonSolver::residual() const { return _residual; }
//-----------------------------------------------------------------------------
double nls::NewtonSolver::residual0() const { return _residual0; }
//-----------------------------------------------------------------------------
bool nls::NewtonSolver::converged(const Vec r,
                                  const NonlinearProblem& nonlinear_problem,
                                  std::size_t newton_iteration)
{
  la::PETScVector _r(r);
  _residual = _r.norm(la::Norm::l2);

  // If this is the first iteration step, set initial residual
  if (newton_iteration == 0)
    _residual0 = _residual;

  // Relative residual
  const double relative_residual = _residual / _residual0;

  // Output iteration number and residual
  if (report && _mpi_comm.rank() == 0)
  {
    spdlog::info("Newton iteration %d: r (abs) = %.3e (tol = %.3e) r (rel) = "
                 "%.3e (tol "
                 "= %.3e)",
                 newton_iteration, _residual, atol, relative_residual, rtol);
  }

  // Return true if convergence criterion is met
  if (relative_residual < rtol or _residual < atol)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
void nls::NewtonSolver::update_solution(
    Vec x, const Vec dx, double relaxation,
    const NonlinearProblem& nonlinear_problem, std::size_t interation)
{
  VecAXPY(x, -relaxation, dx);
}
//-----------------------------------------------------------------------------
