// Copyright (C) 2005-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// #ifdef HAS_PETSC

#include "NewtonSolver.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/petsc.h>
#include <string>
#include <utility>

using namespace dolfinx;

/// Convergence test
/// @param solver The Newton solver
/// @param r The residual vector
/// @return The pair `(residual norm, converged)`, where `converged` is
/// and true` if convergence achieved
std::pair<double, bool> NewtonSolver::converged(const Vec r)
{
  PetscReal residual = 0;
  VecNorm(r, NORM_2, &residual);

  // Relative residual
  const double relative_residual = residual / residual0();

  // Output iteration number and residual
  if (dolfinx::MPI::rank(comm()) == 0)
  {
    spdlog::info("Newton iteration {}"
                 ": r (abs) = {} (tol = {}), r (rel) = {} (tol = {})",
                 _iteration, residual, atol, relative_residual, rtol);
  }

  // Return true if convergence criterion is met
  if (relative_residual < rtol or residual < atol)
    return {residual, true};
  else
    return {residual, false};
}

//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(MPI_Comm comm)
    : _krylov_iterations(0), _iteration(0), _residual(0), _residual0(0),
      _solver(comm), _dx(nullptr), _comm(comm)
{
  // Create linear solver if not already created. Default to LU.
  _solver.set_options_prefix("nls_solve_");
  la::petsc::options::set("nls_solve_ksp_type", "preonly");
  la::petsc::options::set("nls_solve_pc_type", "lu");
  _solver.set_from_options();
}
//-----------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
  if (_b)
    VecDestroy(&_b);
  if (_dx)
    VecDestroy(&_dx);
  if (_matJ)
    MatDestroy(&_matJ);
  if (_matP)
    MatDestroy(&_matP);
}
//-----------------------------------------------------------------------------
void NewtonSolver::setF(std::function<void(const Vec, Vec)> F,
                                    Vec b)
{
  _fnF = std::move(F);
  _b = b;
  PetscObjectReference((PetscObject)_b);
}
//-----------------------------------------------------------------------------
void NewtonSolver::setJ(std::function<void(const Vec, Mat)> J,
                                    Mat Jmat)
{
  _fnJ = std::move(J);
  _matJ = Jmat;
  PetscObjectReference((PetscObject)_matJ);
}
//-----------------------------------------------------------------------------
const la::petsc::KrylovSolver&
NewtonSolver::get_krylov_solver() const
{
  return _solver;
}
//-----------------------------------------------------------------------------
la::petsc::KrylovSolver& NewtonSolver::get_krylov_solver()
{
  return _solver;
}
//-----------------------------------------------------------------------------
void NewtonSolver::set_form(std::function<void(Vec)> form)
{
  _system = std::move(form);
}
//-----------------------------------------------------------------------------
std::pair<int, bool> NewtonSolver::solve(Vec x)
{
  // Reset iteration counts
  _iteration = 0;
  int krylov_iterations = 0;
  _residual = -1;
  _residual0 = 0;

  if (!_fnF)
  {
    throw std::runtime_error("Function for computing residual vector has not "
                             "been provided to the NewtonSolver.");
  }

  if (!_fnJ)
  {
    throw std::runtime_error("Function for computing Jacobian has not "
                             "been provided to the NewtonSolver.");
  }

  if (_system)
    _system(x);
  assert(_b);
  _fnF(x, _b);

  // Check convergence
  bool newton_converged = false;
  if (convergence_criterion == "residual")
    std::tie(_residual, newton_converged) = converged(_b);

  _solver.set_operators(_matJ, _matJ);

  if (!_dx)
    MatCreateVecs(_matJ, &_dx, nullptr);

  // Start iterations
  while (!newton_converged and _iteration < max_it)
  {
    // Compute Jacobian
    assert(_matJ);
    _fnJ(x, _matJ);

    // Perform linear solve and update total number of Krylov iterations
    krylov_iterations += _solver.solve(_dx, _b);

    // Update solution
    VecAXPY(x, -relaxation_parameter, _dx);

    // Increment iteration count
    ++_iteration;

    // Compute F
    if (_system)
      _system(x);
    _fnF(x, _b);
    // Initialize _residual0
    if (_iteration == 1)
    {
      PetscReal _r = 0;
      VecNorm(_dx, NORM_2, &_r);
      _residual0 = _r;
    }

    // Test for convergence
    std::tie(_residual, newton_converged) = converged(_b);
  }

  if (newton_converged)
  {
    if (dolfinx::MPI::rank(_comm.comm()) == 0)
    {
      spdlog::info("Newton solver finished in {} iterations and {} linear "
                   "solver iterations.",
                   _iteration, krylov_iterations);
    }
  }
  else
  {
    throw std::runtime_error("Newton solver did not converge.");
  }

  return {_iteration, newton_converged};
}
//-----------------------------------------------------------------------------
int NewtonSolver::krylov_iterations() const
{
  return _krylov_iterations;
}
//-----------------------------------------------------------------------------
// int NewtonSolver::iteration() const { return _iteration; }
//-----------------------------------------------------------------------------
// double NewtonSolver::residual() const { return _residual; }
//-----------------------------------------------------------------------------
double NewtonSolver::residual0() const { return _residual0; }
//-----------------------------------------------------------------------------
MPI_Comm NewtonSolver::comm() const { return _comm.comm(); }
//-----------------------------------------------------------------------------
// #endif
