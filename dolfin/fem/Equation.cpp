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
// First added:  2011-06-21
// Last changed: 2011-06-21

#include <dolfin/log/log.h>
#include "Form.h"
#include "Equation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Equation::Equation(boost::shared_ptr<const Form> a,
                   boost::shared_ptr<const Form> L)
  : _lhs(a), _rhs(L), _is_linear(true)
{
  assert(a);
  assert(L);

  // Check rank of bilinear form a
  if (a->rank() != 2)
    dolfin_error("Equation.cpp",
                 "define linear variational problem a == L",
                 "expecting the left-hand side to be a bilinear form (not rank %d).",
                 a->rank());

  // Check rank of linear form L
  if (L->rank() != 1)
    dolfin_error("Equation.cpp",
                 "define linear variational problem a == L",
                 "expecting the right-hand side to be a linear form (not rank %d).",
                 a->rank());
}
//-----------------------------------------------------------------------------
Equation::Equation(boost::shared_ptr<const Form> F, int rhs)
  : _lhs(F), _is_linear(false)
{
  assert(F);

  // Check rank of residual F
  if (F->rank() != 1)
    dolfin_error("Equation.cpp",
                 "define nonlinear variational problem F == 0",
                 "expecting the left-hand side to be a linear form (not rank %d).",
                 F->rank());

  // Check value of right-hand side
  if (rhs != 0)
    dolfin_error("Equation.cpp",
                 "define nonlinear variational problem F == 0",
                 "expecting the right-hand side to be zero");
}
//-----------------------------------------------------------------------------
Equation::~Equation()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool Equation::is_linear() const
{
  return _is_linear;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const Form> Equation::lhs() const
{
  assert(_lhs);
  return _lhs;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const Form> Equation::rhs() const
{
  assert(_rhs);
  return _rhs;
}
//-----------------------------------------------------------------------------
