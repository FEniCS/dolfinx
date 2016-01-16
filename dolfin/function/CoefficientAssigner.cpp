// Copyright (C) 2008-2009 Anders Logg
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
// Modified by Martin Alnes, 2008.

#include <memory>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/fem/Form.h>
#include <dolfin/log/log.h>
#include "CoefficientAssigner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CoefficientAssigner::CoefficientAssigner(Form& form, std::size_t number)
  : _form(form), _number(number)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CoefficientAssigner::~CoefficientAssigner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void CoefficientAssigner::operator= (std::shared_ptr<const GenericFunction> coefficient)
{
  dolfin_assert(coefficient);
  _form.set_coefficient(_number, coefficient);
}
//-----------------------------------------------------------------------------
