// Copyright (C) 2011 Anders Logg
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
// Modified by Corrado Maurini, 2013.

#include <dolfin/function/Function.h>
#include "Form.h"
#include "NonlinearVariationalProblem.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NonlinearVariationalProblem::NonlinearVariationalProblem(
  std::shared_ptr<const Form> F,
  std::shared_ptr<Function> u,
  std::vector<std::shared_ptr<const DirichletBC>> bcs)
  : Hierarchical<NonlinearVariationalProblem>(*this), _residual(F), _u(u),
  _bcs(bcs)
{
  // Check forms
  check_forms();
}
//-----------------------------------------------------------------------------
NonlinearVariationalProblem::NonlinearVariationalProblem(
  std::shared_ptr<const Form> F,
  std::shared_ptr<Function> u,
  std::vector<std::shared_ptr<const DirichletBC>> bcs,
  std::shared_ptr<const Form> J)
  : Hierarchical<NonlinearVariationalProblem>(*this), _residual(F),
  _jacobian(J), _u(u), _bcs(bcs)
{
  // Check forms
  check_forms();
}
//-----------------------------------------------------------------------------
void NonlinearVariationalProblem::set_bounds(const Function& lb_func,
                                             const Function& ub_func)
{
    set_bounds(lb_func.vector(), ub_func.vector());
}
//-----------------------------------------------------------------------------
void NonlinearVariationalProblem::set_bounds(
  std::shared_ptr<const GenericVector> lb,
  std::shared_ptr<const GenericVector> ub)
{
    this->_lb = lb;
    this->_ub = ub;
    dolfin_assert(_lb);
    dolfin_assert(_ub);
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
std::shared_ptr<Function> NonlinearVariationalProblem::solution()
{
  return _u;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Function> NonlinearVariationalProblem::solution() const
{
  return _u;
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const DirichletBC>>
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
std::shared_ptr<const GenericVector>
NonlinearVariationalProblem::lower_bound() const
{
  dolfin_assert(_lb);
  return _lb;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const GenericVector>
NonlinearVariationalProblem::upper_bound() const
{
  dolfin_assert(_ub);
  return _ub;
}
//-----------------------------------------------------------------------------
bool NonlinearVariationalProblem::has_jacobian() const
{
  return _jacobian ?  true : false;
}
//-----------------------------------------------------------------------------
bool NonlinearVariationalProblem::has_lower_bound() const
{
  return _lb ?  true : false;
}
//-----------------------------------------------------------------------------
bool NonlinearVariationalProblem::has_upper_bound() const
{
  return _ub ?  true : false;
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
}
//-----------------------------------------------------------------------------
