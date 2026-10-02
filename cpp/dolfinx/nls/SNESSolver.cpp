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
#include <dolfinx/common/petsc.h>
#include <exception>
#include <petscsys.h>
#include <petscsystypes.h>
#include <stdexcept>
#include <string>
#include <utility>

using namespace dolfinx;

namespace
{
// Run a callback, catching and storing any exception it throws (for
// solve() to re-throw) into `exception` and returning a PETSc error
// code in its place, since a C++ exception cannot propagate through
// the PETSc C callback frames.
template <typename F>
PetscErrorCode invoke(F&& f, std::exception_ptr& exception)
{
  try
  {
    std::forward<F>(f)();
    return PETSC_SUCCESS;
  }
  catch (const std::exception& e)
  {
    // Logging matters when the caller ran SNESSolve directly, as
    // nothing then re-throws `exception`
    spdlog::error("Exception raised in a SNES callback: {}", e.what());
    exception = std::current_exception();
    return PETSC_ERR_LIB;
  }
  catch (...)
  {
    spdlog::error("Unknown exception raised in a SNES callback.");
    exception = std::current_exception();
    return PETSC_ERR_LIB;
  }
}
} // namespace

//-----------------------------------------------------------------------------
nls::petsc::SNESSolver::SNESSolver(MPI_Comm comm) : _snes(nullptr)
{
  common::petsc::check(SNESCreate(comm, &_snes), "SNESCreate");
  set_callbacks();
}
//-----------------------------------------------------------------------------
nls::petsc::SNESSolver::SNESSolver(SNES snes, bool inc_ref_count) : _snes(snes)
{
  if (!_snes)
    throw std::runtime_error("PETSc SNES must be initialised before wrapping");

  if (inc_ref_count)
  {
    common::petsc::check(
        PetscObjectReference(reinterpret_cast<PetscObject>(_snes)),
        "PetscObjectReference");
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
    common::petsc::check(VecDestroy(&_b), "VecDestroy");
  if (_matJ)
    common::petsc::check(MatDestroy(&_matJ), "MatDestroy");
  if (_matP)
    common::petsc::check(MatDestroy(&_matP), "MatDestroy");
  if (_snes)
    common::petsc::check(SNESDestroy(&_snes), "SNESDestroy");
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

  common::petsc::check(
      PetscObjectReference(reinterpret_cast<PetscObject>(b_layout)),
      "PetscObjectReference");
  if (_b)
    common::petsc::check(VecDestroy(&_b), "VecDestroy");
  _b = b_layout;

  common::petsc::check(SNESSetFunction(_snes, _b, residual, this),
                       "SNESSetFunction");
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

  common::petsc::check(
      PetscObjectReference(reinterpret_cast<PetscObject>(J_layout)),
      "PetscObjectReference");
  common::petsc::check(
      PetscObjectReference(reinterpret_cast<PetscObject>(P_layout)),
      "PetscObjectReference");

  if (_matJ)
    common::petsc::check(MatDestroy(&_matJ), "MatDestroy");
  if (_matP)
    common::petsc::check(MatDestroy(&_matP), "MatDestroy");
  _matJ = J_layout;
  _matP = P_layout;

  common::petsc::check(SNESSetJacobian(_snes, _matJ, _matP, jacobian, this),
                       "SNESSetJacobian");
}
//-----------------------------------------------------------------------------
void nls::petsc::SNESSolver::set_update(
    std::function<void(PetscInt step)> update)
{
  assert(_snes);
  _fnupdate = std::move(update);
  common::petsc::check(SNESSetUpdate(_snes, update_step), "SNESSetUpdate");
}
//-----------------------------------------------------------------------------
SNESConvergedReason nls::petsc::SNESSolver::solve(Vec x)
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

  common::petsc::check(ierr, "SNESSolve");

  PetscInt num_iterations = 0;
  common::petsc::check(SNESGetIterationNumber(_snes, &num_iterations),
                       "SNESGetIterationNumber");

  // Check if the solution converged and warn if not. Note: this does
  // not throw on non-convergence -- the caller must check the
  // returned convergence reason if this matters for their use case.
  SNESConvergedReason reason;
  common::petsc::check(SNESGetConvergedReason(_snes, &reason),
                       "SNESGetConvergedReason");
  if (reason < 0)
  {
    const char* reason_str;
    common::petsc::check(SNESGetConvergedReasonString(_snes, &reason_str),
                         "SNESGetConvergedReasonString");
    spdlog::warn("PETSc SNES solver did not converge in {} iterations "
                 "(PETSc reason: {}).",
                 num_iterations, reason_str);
  }

  return reason;
}
//-----------------------------------------------------------------------------
void nls::petsc::SNESSolver::set_options_prefix(std::string_view options_prefix)
{
  assert(_snes);
  common::petsc::check(
      SNESSetOptionsPrefix(_snes, std::string(options_prefix).c_str()),
      "SNESSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string nls::petsc::SNESSolver::get_options_prefix() const
{
  assert(_snes);
  const char* prefix = nullptr;
  common::petsc::check(SNESGetOptionsPrefix(_snes, &prefix),
                       "SNESGetOptionsPrefix");
  // PETSc reports an unset prefix as a null pointer
  return prefix ? std::string(prefix) : std::string();
}
//-----------------------------------------------------------------------------
void nls::petsc::SNESSolver::set_from_options() const
{
  assert(_snes);
  common::petsc::check(SNESSetFromOptions(_snes), "SNESSetFromOptions");
}
//-----------------------------------------------------------------------------
SNES nls::petsc::SNESSolver::snes() const { return _snes; }
//-----------------------------------------------------------------------------
PetscErrorCode nls::petsc::SNESSolver::residual(SNES, Vec x, Vec b, void* ctx)
{
  SNESSolver* solver = static_cast<SNESSolver*>(ctx);
  assert(solver->_fnF);
  return invoke([&solver, &x, &b] { solver->_fnF(x, b); }, solver->_exception);
}
//-----------------------------------------------------------------------------
PetscErrorCode nls::petsc::SNESSolver::jacobian(SNES, Vec x, Mat Jmat, Mat Pmat,
                                                void* ctx)
{
  SNESSolver* solver = static_cast<SNESSolver*>(ctx);
  assert(solver->_fnJ);
  return invoke([&solver, &x, &Jmat, &Pmat] { solver->_fnJ(x, Jmat, Pmat); },
                solver->_exception);
}
//-----------------------------------------------------------------------------
PetscErrorCode nls::petsc::SNESSolver::update_step(SNES snes, PetscInt step)
{
  // SNESSetUpdate takes no context argument, so recover the solver
  // from the SNES application context that set_callbacks() claimed.
  // Errors are returned rather than thrown, as this is called from
  // PETSc.
  SNESSolver* solver = nullptr;
  PetscErrorCode ierr
      = SNESGetApplicationContext(snes, static_cast<void*>(&solver));
  if (ierr != 0)
    return ierr;

  assert(solver);
  assert(solver->_fnupdate);
  return invoke([&solver, &step] { solver->_fnupdate(step); },
                solver->_exception);
}
//-----------------------------------------------------------------------------
void nls::petsc::SNESSolver::set_callbacks()
{
  if (!_snes)
    return;

  common::petsc::check(SNESSetApplicationContext(_snes, this),
                       "SNESSetApplicationContext");

  if (_fnF)
  {
    common::petsc::check(SNESSetFunction(_snes, _b, residual, this),
                         "SNESSetFunction");
  }

  if (_fnJ)
  {
    common::petsc::check(SNESSetJacobian(_snes, _matJ, _matP, jacobian, this),
                         "SNESSetJacobian");
  }

  if (_fnupdate)
  {
    common::petsc::check(SNESSetUpdate(_snes, update_step), "SNESSetUpdate");
  }
}
//-----------------------------------------------------------------------------

#endif
