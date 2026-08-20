// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_PETSC

#include "NonlinearProblem.h"
#include <cassert>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/petsc.h>
#include <exception>
#include <petscsystypes.h>
#include <string>
#include <utility>

using namespace dolfinx;

//-----------------------------------------------------------------------------
nls::petsc::NonlinearProblem::NonlinearProblem(MPI_Comm comm) : _snes(nullptr)
{
  PetscErrorCode ierr = SNESCreate(comm, &_snes);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "SNESCreate");
}
//-----------------------------------------------------------------------------
nls::petsc::NonlinearProblem::NonlinearProblem(SNES snes, bool inc_ref_count)
    : _snes(snes)
{
  assert(_snes);
  if (inc_ref_count)
  {
    PetscErrorCode ierr = PetscObjectReference((PetscObject)_snes);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "PetscObjectReference");
  }
}
//-----------------------------------------------------------------------------
nls::petsc::NonlinearProblem::NonlinearProblem(
    NonlinearProblem&& problem) noexcept
    : _fnF(std::move(problem._fnF)), _fnJ(std::move(problem._fnJ)),
      _b(std::exchange(problem._b, nullptr)),
      _matJ(std::exchange(problem._matJ, nullptr)),
      _matP(std::exchange(problem._matP, nullptr)),
      _snes(std::exchange(problem._snes, nullptr))
{
  set_callbacks();
}
//-----------------------------------------------------------------------------
nls::petsc::NonlinearProblem::~NonlinearProblem()
{
  if (_b)
    VecDestroy(&_b);
  if (_matJ)
    MatDestroy(&_matJ);
  if (_matP)
    MatDestroy(&_matP);
  if (_snes)
    SNESDestroy(&_snes);
}
//-----------------------------------------------------------------------------
nls::petsc::NonlinearProblem&
nls::petsc::NonlinearProblem::operator=(NonlinearProblem&& problem) noexcept
{
  std::swap(_fnF, problem._fnF);
  std::swap(_fnJ, problem._fnJ);
  std::swap(_b, problem._b);
  std::swap(_matJ, problem._matJ);
  std::swap(_matP, problem._matP);
  std::swap(_snes, problem._snes);

  set_callbacks();
  problem.set_callbacks();

  return *this;
}
//-----------------------------------------------------------------------------
void nls::petsc::NonlinearProblem::set_F(std::function<void(const Vec, Vec)> F,
                                         Vec b)
{
  assert(_snes);
  assert(b);
  _fnF = std::move(F);

  PetscErrorCode ierr = PetscObjectReference((PetscObject)b);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "PetscObjectReference");
  if (_b)
    VecDestroy(&_b);
  _b = b;

  ierr = SNESSetFunction(_snes, _b, residual, this);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "SNESSetFunction");
}
//-----------------------------------------------------------------------------
void nls::petsc::NonlinearProblem::set_J(
    std::function<void(const Vec, Mat, Mat)> J, Mat Jmat, Mat Pmat)
{
  assert(_snes);
  assert(Jmat);
  _fnJ = std::move(J);

  if (!Pmat)
    Pmat = Jmat;

  PetscErrorCode ierr = PetscObjectReference((PetscObject)Jmat);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "PetscObjectReference");
  ierr = PetscObjectReference((PetscObject)Pmat);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "PetscObjectReference");

  if (_matJ)
    MatDestroy(&_matJ);
  if (_matP)
    MatDestroy(&_matP);
  _matJ = Jmat;
  _matP = Pmat;

  ierr = SNESSetJacobian(_snes, _matJ, _matP, jacobian, this);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "SNESSetJacobian");
}
//-----------------------------------------------------------------------------
int nls::petsc::NonlinearProblem::solve(Vec x) const
{
  common::Timer timer("PETSc SNES solver");
  assert(_snes);
  assert(x);

  spdlog::info("PETSc SNES solver starting to solve system.");
  PetscErrorCode ierr = SNESSolve(_snes, nullptr, x);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "SNESSolve");

  PetscInt num_iterations = 0;
  ierr = SNESGetIterationNumber(_snes, &num_iterations);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "SNESGetIterationNumber");

  // Check if the solution converged and warn if not. Note: this does
  // not throw on non-convergence -- the caller is responsible for
  // checking the convergence reason (via snes()) if this matters for
  // its use case.
  SNESConvergedReason reason;
  ierr = SNESGetConvergedReason(_snes, &reason);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "SNESGetConvergedReason");
  if (reason < 0)
  {
    const char* reason_str;
    ierr = SNESGetConvergedReasonString(_snes, &reason_str);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "SNESGetConvergedReasonString");
    spdlog::warn("PETSc SNES solver did not converge in {} iterations "
                 "(PETSc reason: {}).",
                 num_iterations, reason_str);
  }

  return num_iterations;
}
//-----------------------------------------------------------------------------
void nls::petsc::NonlinearProblem::set_options_prefix(
    std::string_view options_prefix)
{
  assert(_snes);
  PetscErrorCode ierr
      = SNESSetOptionsPrefix(_snes, std::string(options_prefix).c_str());
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "SNESSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string nls::petsc::NonlinearProblem::get_options_prefix() const
{
  assert(_snes);
  const char* prefix = nullptr;
  PetscErrorCode ierr = SNESGetOptionsPrefix(_snes, &prefix);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "SNESGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void nls::petsc::NonlinearProblem::set_from_options() const
{
  assert(_snes);
  PetscErrorCode ierr = SNESSetFromOptions(_snes);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "SNESSetFromOptions");
}
//-----------------------------------------------------------------------------
SNES nls::petsc::NonlinearProblem::snes() const { return _snes; }
//-----------------------------------------------------------------------------
PetscErrorCode nls::petsc::NonlinearProblem::residual(SNES, Vec x, Vec b,
                                                      void* ctx)
{
  NonlinearProblem* problem = static_cast<NonlinearProblem*>(ctx);
  assert(problem->_fnF);
  try
  {
    problem->_fnF(x, b);
  }
  catch (const std::exception& e)
  {
    // An exception cannot be propagated through the PETSc C frames
    spdlog::error("Exception raised in SNES residual callback: {}", e.what());
    return PETSC_ERR_LIB;
  }

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode nls::petsc::NonlinearProblem::jacobian(SNES, Vec x, Mat Jmat,
                                                      Mat Pmat, void* ctx)
{
  NonlinearProblem* problem = static_cast<NonlinearProblem*>(ctx);
  assert(problem->_fnJ);
  try
  {
    problem->_fnJ(x, Jmat, Pmat);
  }
  catch (const std::exception& e)
  {
    // An exception cannot be propagated through the PETSc C frames
    spdlog::error("Exception raised in SNES Jacobian callback: {}", e.what());
    return PETSC_ERR_LIB;
  }

  return 0;
}
//-----------------------------------------------------------------------------
void nls::petsc::NonlinearProblem::set_callbacks()
{
  if (!_snes)
    return;

  PetscErrorCode ierr;
  if (_fnF)
  {
    ierr = SNESSetFunction(_snes, _b, residual, this);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "SNESSetFunction");
  }

  if (_fnJ)
  {
    ierr = SNESSetJacobian(_snes, _matJ, _matP, jacobian, this);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "SNESSetJacobian");
  }
}
//-----------------------------------------------------------------------------

#endif
