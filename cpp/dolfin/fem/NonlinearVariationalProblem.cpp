// Copyright (C) 2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "NonlinearVariationalProblem.h"
#include "DirichletBC.h"
#include "Form.h"
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScVector.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
NonlinearVariationalProblem::NonlinearVariationalProblem(
    std::shared_ptr<const Form> F, std::shared_ptr<Function> u,
    std::vector<std::shared_ptr<const fem::DirichletBC>> bcs,
    std::shared_ptr<const Form> J)
    : _residual(F), _jacobian(J), _u(u), _bcs(bcs)
{
  // Check forms
  check_forms();
}
//-----------------------------------------------------------------------------
void NonlinearVariationalProblem::set_bounds(const Function& lb_func,
                                             const Function& ub_func)
{
  this->set_bounds(lb_func.vector(), ub_func.vector());
}
//-----------------------------------------------------------------------------
void NonlinearVariationalProblem::set_bounds(
    std::shared_ptr<const PETScVector> lb,
    std::shared_ptr<const PETScVector> ub)
{
  dolfin_assert(lb);
  dolfin_assert(ub);
  this->_lb = lb;
  this->_ub = ub;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Form> NonlinearVariationalProblem::residual_form() const
{
  return _residual;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Form> NonlinearVariationalProblem::jacobian_form() const
{
  return _jacobian;
}
//-----------------------------------------------------------------------------
std::shared_ptr<Function> NonlinearVariationalProblem::solution() { return _u; }
//-----------------------------------------------------------------------------
std::shared_ptr<const Function> NonlinearVariationalProblem::solution() const
{
  return _u;
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const fem::DirichletBC>>
NonlinearVariationalProblem::bcs() const
{
  return _bcs;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FunctionSpace>
NonlinearVariationalProblem::trial_space() const
{
  dolfin_assert(_u);
  return _u->function_space();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FunctionSpace>
NonlinearVariationalProblem::test_space() const
{
  dolfin_assert(_residual);
  return _residual->function_space(0);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const PETScVector>
NonlinearVariationalProblem::lower_bound() const
{
  dolfin_assert(_lb);
  return _lb;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const PETScVector>
NonlinearVariationalProblem::upper_bound() const
{
  dolfin_assert(_ub);
  return _ub;
}
//-----------------------------------------------------------------------------
void NonlinearVariationalProblem::check_forms() const
{
  // Check rank of residual F
  dolfin_assert(_residual);
  if (_residual->rank() != 1)
  {
    dolfin_error("NonlinearVariationalProblem.cpp",
                 "define nonlinear variational problem F(u; v) = 0 for all v",
                 "Expecting the residual F to be a linear form (not rank %d)",
                 _residual->rank());
  }

  // Check rank of Jacobian J
  if (_jacobian && _jacobian->rank() != 2)
  {
    dolfin_error("NonlinearVariationalProblem.cpp",
                 "define nonlinear variational problem F(u; v) = 0 for all v",
                 "Expecting the Jacobian J to be a bilinear form (not rank %d)",
                 _jacobian->rank());
  }

  // FIXME: Should we add a check here that matches the function space
  // FIXME: of the solution variable u to a coefficient space for F?
  dolfin_assert(_u);

  // Check that function spaces of bcs are contained in trial space
  dolfin_assert(_u);
  const auto trial_space = _u->function_space();
  dolfin_assert(trial_space);
  for (const auto bc : _bcs)
  {
    dolfin_assert(bc);
    const auto bc_space = bc->function_space();
    dolfin_assert(bc_space);
    if (!trial_space->contains(*bc_space))
    {
      dolfin_error("NonlinearVariationalProblem.cpp",
                   "define nonlinear variational problem F(u; v) = 0 for all v",
                   "Expecting the boundary conditions to to live on (a "
                   "subspace of) the trial space");
    }
  }
}
//-----------------------------------------------------------------------------
