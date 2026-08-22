// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_PETSC

#include "SNESSolver.h"
#include <cassert>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/petsc.h>
#include <exception>
#include <petscsys.h>
#include <petscsystypes.h>
#include <string>
#include <utility>

using namespace dolfinx;

//-----------------------------------------------------------------------------
// Check a PETSc error code and throw a descriptive exception if it is
// non-zero. Expects a local `PetscErrorCode ierr` in scope.
#define CHECK_ERROR(NAME)                                                      \
  do                                                                           \
  {                                                                            \
    if (ierr != 0)                                                             \
      la::petsc::error(ierr, __FILE__, NAME);                                  \
  } while (0)

//-----------------------------------------------------------------------------
nls::petsc::SNESSolver::SNESSolver(MPI_Comm comm) : _snes(nullptr)
{
  PetscErrorCode ierr = SNESCreate(comm, &_snes);
  CHECK_ERROR("SNESCreate");
  set_callbacks();
}
//-----------------------------------------------------------------------------
nls::petsc::SNESSolver::SNESSolver(SNES snes, bool inc_ref_count) : _snes(snes)
{
  assert(_snes);
  if (inc_ref_count)
  {
    PetscErrorCode ierr
        = PetscObjectReference(reinterpret_cast<PetscObject>(_snes));
    CHECK_ERROR("PetscObjectReference");
  }

  set_callbacks();
}
//-----------------------------------------------------------------------------
nls::petsc::SNESSolver::SNESSolver(SNESSolver&& solver) noexcept
    : _fnF(std::move(solver._fnF)), _fnJ(std::move(solver._fnJ)),
      _fnupdate(std::move(solver._fnupdate)),
      _exception(std::exchange(solver._exception, nullptr)),
      _b(std::exchange(solver._b, nullptr)),
      _matJ(std::exchange(solver._matJ, nullptr)),
      _matP(std::exchange(solver._matP, nullptr)),
      _snes(std::exchange(solver._snes, nullptr))
{
  set_callbacks();
}
//-----------------------------------------------------------------------------
nls::petsc::SNESSolver::~SNESSolver()
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
nls::petsc::SNESSolver&
nls::petsc::SNESSolver::operator=(SNESSolver&& solver) noexcept
{
  std::swap(_fnF, solver._fnF);
  std::swap(_fnJ, solver._fnJ);
  std::swap(_fnupdate, solver._fnupdate);
  std::swap(_exception, solver._exception);
  std::swap(_b, solver._b);
  std::swap(_matJ, solver._matJ);
  std::swap(_matP, solver._matP);
  std::swap(_snes, solver._snes);

  set_callbacks();
  solver.set_callbacks();

  return *this;
}
//-----------------------------------------------------------------------------
void nls::petsc::SNESSolver::set_F(std::function<void(const Vec x, Vec b)> F,
                                   Vec b_layout)
{
  assert(_snes);
  assert(b_layout);
  _fnF = std::move(F);

  PetscErrorCode ierr
      = PetscObjectReference(reinterpret_cast<PetscObject>(b_layout));
  CHECK_ERROR("PetscObjectReference");
  if (_b)
    VecDestroy(&_b);
  _b = b_layout;

  ierr = SNESSetFunction(_snes, _b, residual, this);
  CHECK_ERROR("SNESSetFunction");
}
//-----------------------------------------------------------------------------
void nls::petsc::SNESSolver::set_J(
    std::function<void(const Vec x, Mat Jmat, Mat Pmat)> J, Mat J_layout,
    Mat P_layout)
{
  assert(_snes);
  assert(J_layout);
  _fnJ = std::move(J);

  if (!P_layout)
    P_layout = J_layout;

  PetscErrorCode ierr
      = PetscObjectReference(reinterpret_cast<PetscObject>(J_layout));
  CHECK_ERROR("PetscObjectReference");
  ierr = PetscObjectReference(reinterpret_cast<PetscObject>(P_layout));
  CHECK_ERROR("PetscObjectReference");

  if (_matJ)
    MatDestroy(&_matJ);
  if (_matP)
    MatDestroy(&_matP);
  _matJ = J_layout;
  _matP = P_layout;

  ierr = SNESSetJacobian(_snes, _matJ, _matP, jacobian, this);
  CHECK_ERROR("SNESSetJacobian");
}
//-----------------------------------------------------------------------------
void nls::petsc::SNESSolver::set_update(
    std::function<void(PetscInt step)> update)
{
  assert(_snes);
  _fnupdate = std::move(update);
  PetscErrorCode ierr = SNESSetUpdate(_snes, update_step);
  CHECK_ERROR("SNESSetUpdate");
}
//-----------------------------------------------------------------------------
PetscInt nls::petsc::SNESSolver::solve(Vec x)
{
  common::Timer timer("PETSc SNES solver");
  assert(_snes);
  assert(x);

  spdlog::info("PETSc SNES solver starting to solve system.");

  // Discard an exception from a solve that bypassed this one
  _exception = nullptr;
  PetscErrorCode ierr = SNESSolve(_snes, nullptr, x);

  // A callback cannot throw through PETSc, so it stored its exception
  // and returned an error code. Re-throw in place of that code.
  if (_exception)
    std::rethrow_exception(std::exchange(_exception, nullptr));

  CHECK_ERROR("SNESSolve");

  PetscInt num_iterations = 0;
  ierr = SNESGetIterationNumber(_snes, &num_iterations);
  CHECK_ERROR("SNESGetIterationNumber");

  // Check if the solution converged and warn if not. Note: this does
  // not throw on non-convergence -- the caller is responsible for
  // checking the convergence reason (via snes()) if this matters for
  // their use case.
  SNESConvergedReason reason;
  ierr = SNESGetConvergedReason(_snes, &reason);
  CHECK_ERROR("SNESGetConvergedReason");
  if (reason < 0)
  {
    const char* reason_str;
    ierr = SNESGetConvergedReasonString(_snes, &reason_str);
    CHECK_ERROR("SNESGetConvergedReasonString");
    spdlog::warn("PETSc SNES solver did not converge in {} iterations "
                 "(PETSc reason: {}).",
                 num_iterations, reason_str);
  }

  return num_iterations;
}
//-----------------------------------------------------------------------------
void nls::petsc::SNESSolver::set_options_prefix(std::string_view options_prefix)
{
  assert(_snes);
  PetscErrorCode ierr
      = SNESSetOptionsPrefix(_snes, std::string(options_prefix).c_str());
  CHECK_ERROR("SNESSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string nls::petsc::SNESSolver::get_options_prefix() const
{
  assert(_snes);
  const char* prefix = nullptr;
  PetscErrorCode ierr = SNESGetOptionsPrefix(_snes, &prefix);
  CHECK_ERROR("SNESGetOptionsPrefix");
  // PETSc reports an unset prefix as a null pointer
  return prefix ? std::string(prefix) : std::string();
}
//-----------------------------------------------------------------------------
void nls::petsc::SNESSolver::set_from_options() const
{
  assert(_snes);
  PetscErrorCode ierr = SNESSetFromOptions(_snes);
  CHECK_ERROR("SNESSetFromOptions");
}
//-----------------------------------------------------------------------------
SNES nls::petsc::SNESSolver::snes() const { return _snes; }
//-----------------------------------------------------------------------------
PetscErrorCode nls::petsc::SNESSolver::residual(SNES, Vec x, Vec b, void* ctx)
{
  SNESSolver* solver = static_cast<SNESSolver*>(ctx);
  assert(solver->_fnF);
  try
  {
    solver->_fnF(x, b);
  }
  catch (...)
  {
    return solver->store_exception();
  }

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode nls::petsc::SNESSolver::jacobian(SNES, Vec x, Mat Jmat, Mat Pmat,
                                                void* ctx)
{
  SNESSolver* solver = static_cast<SNESSolver*>(ctx);
  assert(solver->_fnJ);
  try
  {
    solver->_fnJ(x, Jmat, Pmat);
  }
  catch (...)
  {
    return solver->store_exception();
  }

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode nls::petsc::SNESSolver::update_step(SNES snes, PetscInt step)
{
  // SNESSetUpdate takes no context argument, so recover the solver from
  // the context registered with SNESSetFunction. Errors are returned
  // rather than thrown, as this is called from PETSc.
  decltype(&residual) fn = nullptr;
  void* ctx = nullptr;
  PetscErrorCode ierr = SNESGetFunction(snes, nullptr, &fn, &ctx);
  if (ierr != 0)
    return ierr;

  // The context is this class's only if the residual callback is, which
  // it is not if the caller set their own on a wrapped SNES
  if (fn != residual)
  {
    spdlog::error("SNES residual function was not set by SNESSolver, so the "
                  "solver cannot be recovered in the update hook.");
    return PETSC_ERR_ARG_WRONGSTATE;
  }

  SNESSolver* solver = static_cast<SNESSolver*>(ctx);
  assert(solver);

  assert(solver->_fnupdate);
  try
  {
    solver->_fnupdate(step);
  }
  catch (...)
  {
    return solver->store_exception();
  }

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode nls::petsc::SNESSolver::store_exception()
{
  _exception = std::current_exception();

  // Re-throw and catch to read the message, the only way to inspect an
  // exception_ptr. Logging matters when the caller ran SNESSolve
  // directly, as nothing then re-throws.
  try
  {
    std::rethrow_exception(_exception);
  }
  catch (const std::exception& e)
  {
    spdlog::error("Exception raised in a SNES callback: {}", e.what());
  }
  catch (...)
  {
    // An exception_ptr need not hold a std::exception
    spdlog::error("Unknown exception raised in a SNES callback.");
  }

  return PETSC_ERR_LIB;
}
//-----------------------------------------------------------------------------
void nls::petsc::SNESSolver::set_callbacks()
{
  if (!_snes)
    return;

  PetscErrorCode ierr = 0;
  if (_fnF)
  {
    ierr = SNESSetFunction(_snes, _b, residual, this);
    CHECK_ERROR("SNESSetFunction");
  }

  if (_fnJ)
  {
    ierr = SNESSetJacobian(_snes, _matJ, _matP, jacobian, this);
    CHECK_ERROR("SNESSetJacobian");
  }

  if (_fnupdate)
  {
    ierr = SNESSetUpdate(_snes, update_step);
    CHECK_ERROR("SNESSetUpdate");
  }
}
//-----------------------------------------------------------------------------

#endif
