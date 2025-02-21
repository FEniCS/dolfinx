// Copyright (C) 2005-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_PETSC

#include "NewtonSolver.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/petsc.h>
#include <string>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------

/// Convergence test
/// @param solver The Newton solver
/// @param r The residual vector
/// @return The pair `(residual norm, converged)`, where `converged` is
/// and true` if convergence achieved
std::pair<double, bool> converged(const nls::petsc::NewtonSolver& solver,
                                  const Vec r)
{
  PetscReal residual = 0.0;
  VecNorm(r, NORM_2, &residual);

  // Relative residual
  const double relative_residual = residual / solver.residual0();

  // Output iteration number and residual
  if (solver.report and dolfinx::MPI::rank(solver.comm()) == 0)
  {
    spdlog::info("Newton iteration {}"
                 ": r (abs) = {} (tol = {}), r (rel) = {} (tol = {})",
                 solver.iteration(), residual, solver.atol, relative_residual,
                 solver.rtol);
  }

  // Return true if convergence criterion is met
  if (relative_residual < solver.rtol or residual < solver.atol)
    return {residual, true};
  else
    return {residual, false};
}
//-----------------------------------------------------------------------------

/// Update solution vector by computed Newton step. Default update is
/// given by formula::
///
///   x -= relaxation_parameter*dx
///
///  @param solver The Newton solver
///  @param dx The update vector computed by Newton step
///  @param[in,out] x The solution vector to be updated
void update_solution(const nls::petsc::NewtonSolver& solver, const Vec dx,
                     Vec x)
{
  VecAXPY(x, -solver.relaxation_parameter, dx);
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
nls::petsc::NewtonSolver::NewtonSolver(MPI_Comm comm)
    : _converged(converged), _update_solution(update_solution),
      _krylov_iterations(0), _iteration(0), _residual(0.0), _residual0(0.0),
      _solver(comm), _dx(nullptr), _comm(comm)
{
  // Create linear solver if not already created. Default to LU.
  _solver.set_options_prefix("nls_solve_");
  la::petsc::options::set("nls_solve_ksp_type", "preonly");
  la::petsc::options::set("nls_solve_pc_type", "lu");
  _solver.set_from_options();
}
//-----------------------------------------------------------------------------
nls::petsc::NewtonSolver::~NewtonSolver()
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
void nls::petsc::NewtonSolver::setF(std::function<void(const Vec, Vec)> F,
                                    Vec b)
{
  _fnF = F;
  _b = b;
  PetscObjectReference((PetscObject)_b);
}
//-----------------------------------------------------------------------------
void nls::petsc::NewtonSolver::setJ(std::function<void(const Vec, Mat)> J,
                                    Mat Jmat)
{
  _fnJ = J;
  _matJ = Jmat;
  PetscObjectReference((PetscObject)_matJ);
}
//-----------------------------------------------------------------------------
void nls::petsc::NewtonSolver::setP(std::function<void(const Vec, Mat)> P,
                                    Mat Pmat)
{
  _fnP = P;
  _matP = Pmat;
  PetscObjectReference((PetscObject)_matP);
}
//-----------------------------------------------------------------------------
const la::petsc::KrylovSolver&
nls::petsc::NewtonSolver::get_krylov_solver() const
{
  return _solver;
}
//-----------------------------------------------------------------------------
la::petsc::KrylovSolver& nls::petsc::NewtonSolver::get_krylov_solver()
{
  return _solver;
}
//-----------------------------------------------------------------------------
void nls::petsc::NewtonSolver::set_form(std::function<void(Vec)> form)
{
  _system = form;
}
//-----------------------------------------------------------------------------
void nls::petsc::NewtonSolver::set_convergence_check(
    std::function<std::pair<double, bool>(const NewtonSolver&, const Vec)> c)
{
  _converged = c;
}
//-----------------------------------------------------------------------------
void nls::petsc::NewtonSolver::set_update(
    std::function<void(const NewtonSolver& solver, const Vec, Vec)> update)
{
  _update_solution = update;
}
//-----------------------------------------------------------------------------
std::pair<int, bool> nls::petsc::NewtonSolver::solve(Vec x)
{
  // Reset iteration counts
  _iteration = 0;
  _krylov_iterations = 0;
  _residual = -1;

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
  {
    std::tie(_residual, newton_converged) = this->_converged(*this, _b);
    _residual0 = _residual;
  }
  else if (convergence_criterion == "incremental")
  {
    // We need to do at least one Newton step with the ||dx||-stopping
    // criterion
    newton_converged = false;
  }
  else
  {
    throw std::runtime_error("Unknown convergence criterion: "
                             + convergence_criterion);
  }

  // FIXME: check that this is efficient if A and/or P are unchanged
  // Set operators
  if (_matP)
    _solver.set_operators(_matJ, _matP);
  else
    _solver.set_operators(_matJ, _matJ);

  if (!_dx)
    MatCreateVecs(_matJ, &_dx, nullptr);

  // Start iterations
  while (!newton_converged and _iteration < max_it)
  {
    // Compute Jacobian
    assert(_matJ);
    _fnJ(x, _matJ);

    if (_fnP)
      _fnP(x, _matP);

    // Perform linear solve and update total number of Krylov iterations
    _krylov_iterations += _solver.solve(_dx, _b);

    // Update solution
    this->_update_solution(*this, _dx, x);

    // Increment iteration count
    ++_iteration;

    // Compute F
    if (_system)
      _system(x);
    _fnF(x, _b);

    // Test for convergence
    if (convergence_criterion == "residual")
      std::tie(_residual, newton_converged) = this->_converged(*this, _b);
    else if (convergence_criterion == "incremental")
    {
      // Subtract 1 to make sure that the initial residual0 is properly
      // set.
      if (_iteration == 1)
      {
        PetscReal _r = 0.0;
        VecNorm(_dx, NORM_2, &_r);
        _residual0 = _r;
        _residual = 1.0;
        newton_converged = false;
      }
      else
        std::tie(_residual, newton_converged) = this->_converged(*this, _dx);
    }
    else
      throw std::runtime_error("Unknown convergence criterion string.");
  }

  if (newton_converged)
  {
    if (dolfinx::MPI::rank(_comm.comm()) == 0)
    {
      spdlog::info("Newton solver finished in {} iterations and {} linear "
                   "solver iterations.",
                   _iteration, _krylov_iterations);
    }
  }
  else
  {
    if (error_on_nonconvergence)
    {
      if (_iteration == max_it)
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

  return {_iteration, newton_converged};
}
//-----------------------------------------------------------------------------
int nls::petsc::NewtonSolver::krylov_iterations() const
{
  return _krylov_iterations;
}
//-----------------------------------------------------------------------------
int nls::petsc::NewtonSolver::iteration() const { return _iteration; }
//-----------------------------------------------------------------------------
double nls::petsc::NewtonSolver::residual() const { return _residual; }
//-----------------------------------------------------------------------------
double nls::petsc::NewtonSolver::residual0() const { return _residual0; }
//-----------------------------------------------------------------------------
MPI_Comm nls::petsc::NewtonSolver::comm() const { return _comm.comm(); }
//-----------------------------------------------------------------------------
#endif
