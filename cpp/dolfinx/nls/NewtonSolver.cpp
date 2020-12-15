// Copyright (C) 2005-2018 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "NewtonSolver.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/PETScKrylovSolver.h>
#include <dolfinx/la/PETScOptions.h>
#include <dolfinx/la/PETScVector.h>
#include <string>

using namespace dolfinx;

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
void nls::NewtonSolver::setF(const std::function<void(const Vec, Vec)>& F,
                             Vec b)
{
  _fnF = F;
  _b = b;
  PetscObjectReference((PetscObject)_b);
}
//-----------------------------------------------------------------------------
void nls::NewtonSolver::setJ(const std::function<void(const Vec, Mat)>& J,
                             Mat Jmat)
{
  _fnJ = J;
  _matJ = Jmat;
  PetscObjectReference((PetscObject)_matJ);
}
//-----------------------------------------------------------------------------
void nls::NewtonSolver::setP(const std::function<void(const Vec, Mat)>& P,
                             Mat Pmat)
{
  _fnP = P;
  _matP = Pmat;
  PetscObjectReference((PetscObject)_matP);
}
//-----------------------------------------------------------------------------
void nls::NewtonSolver::set_form(const std::function<void(Vec)>& form)
{
  _system = form;
}
//-----------------------------------------------------------------------------
std::pair<int, bool> dolfinx::nls::NewtonSolver::solve(Vec x)
{
  // Reset iteration counts
  int newton_iteration = 0;
  _krylov_iterations = 0;

  if (!_fnF)
  {
    throw std::runtime_error("Function for computing residual vector has not "
                             "been provided to the NewtonSolver.");
  }

  if (!_fnJ)
  {
    throw std::runtime_error("Function for computing Jacobianhas not "
                             "been provided to the NewtonSolver.");
  }

  if (_system)
    _system(x);
  assert(_b);
  _fnF(x, _b);

  // Check convergence
  bool newton_converged = false;
  if (convergence_criterion == "residual")
    newton_converged = converged(_b, 0);
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
  while (!newton_converged and newton_iteration < max_it)
  {
    // Compute Jacobian
    assert(_matJ);
    _fnJ(x, _matJ);

    if (_fnP)
      _fnP(x, _matP);

    // Perform linear solve and update total number of Krylov iterations
    _krylov_iterations += _solver.solve(_dx, _b);

    // Update solution
    update_solution(x, _dx, relaxation_parameter, newton_iteration);

    // Increment iteration count
    ++newton_iteration;

    // FIXME: This step is not needed if residual is based on dx and
    //        this has converged.
    // FIXME: But, this function call may update internal variables, etc.
    // Compute F
    if (_system)
      _system(x);
    _fnF(x, _b);

    // Test for convergence
    if (convergence_criterion == "residual")
      newton_converged = converged(_b, newton_iteration);
    else if (convergence_criterion == "incremental")
    {
      // Subtract 1 to make sure that the initial residual0 is
      // properly set.
      newton_converged = converged(_dx, newton_iteration - 1);
    }
    else
      throw std::runtime_error("Unknown convergence criterion string.");
  }

  if (newton_converged)
  {
    if (dolfinx::MPI::rank(_mpi_comm.comm()) == 0)
    {
      LOG(INFO) << "Newton solver finished in " << newton_iteration
                << " iterations and " << _krylov_iterations
                << " linear solver iterations.";
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
      LOG(WARNING) << "Newton solver did not converge.";
  }

  return std::pair(newton_iteration, newton_converged);
}
//-----------------------------------------------------------------------------
int nls::NewtonSolver::krylov_iterations() const { return _krylov_iterations; }
//-----------------------------------------------------------------------------
double nls::NewtonSolver::residual() const { return _residual; }
//-----------------------------------------------------------------------------
double nls::NewtonSolver::residual0() const { return _residual0; }
//-----------------------------------------------------------------------------
bool nls::NewtonSolver::converged(const Vec r, int iteration)
{
  la::PETScVector _r(r, true);
  _residual = _r.norm(la::Norm::l2);

  // If this is the first iteration step, set initial residual
  if (iteration == 0)
    _residual0 = _residual;

  // Relative residual
  const double relative_residual = _residual / _residual0;

  // Output iteration number and residual
  if (report and dolfinx::MPI::rank(_mpi_comm.comm()) == 0)
  {
    LOG(INFO) << "Newton iteration " << iteration << ": r (abs) = " << _residual
              << " (tol = " << atol << ") r (rel) = " << relative_residual
              << "(tol = " << rtol << ")";
  }

  // Return true if convergence criterion is met
  if (relative_residual < rtol or _residual < atol)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
void nls::NewtonSolver::update_solution(Vec x, const Vec dx, double relaxation,
                                        int)
{
  VecAXPY(x, -relaxation, dx);
}
//-----------------------------------------------------------------------------
