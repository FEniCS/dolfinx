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
// Last changed: 2011-06-22

#include <dolfin/log/log.h>
#include "Form.h"
#include "Equation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Equation::Equation(std::shared_ptr<const Form> a,
                   std::shared_ptr<const Form> L)
  : _lhs(a), _rhs(L), _rhs_int(0), _is_linear(true)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Equation::Equation(std::shared_ptr<const Form> F, int rhs)
  : _lhs(F), _rhs_int(rhs), _is_linear(false)
{
  // Do nothing
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
std::shared_ptr<const Form> Equation::lhs() const
{
  dolfin_assert(_lhs);
  return _lhs;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Form> Equation::rhs() const
{
  dolfin_assert(_rhs);
  return _rhs;
}
//-----------------------------------------------------------------------------
int Equation::rhs_int() const
{
  return _rhs_int;
}
//-----------------------------------------------------------------------------
