// Copyright (C) 2015-2016 Anders Logg
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
// First added:  2015-11-05
// Last changed: 2016-03-02

#include <memory>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Function.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/MultiMeshForm.h>
#include "MultiMeshCoefficientAssigner.h"
#include "MultiMeshFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMeshCoefficientAssigner::MultiMeshCoefficientAssigner
(MultiMeshForm& form, std::size_t number)
  : _form(form), _number(number)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MultiMeshCoefficientAssigner::~MultiMeshCoefficientAssigner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MultiMeshCoefficientAssigner::operator=
(std::shared_ptr<const GenericFunction> coefficient)
{
  // Assign to all parts of form
  for (std::size_t part = 0; part < _form.num_parts(); part++)
  {
    Form& a = const_cast<Form&>(*_form.part(part));
    a.set_coefficient(_number, coefficient);
  }
}
//-----------------------------------------------------------------------------
void MultiMeshCoefficientAssigner::operator=
(std::shared_ptr<const MultiMeshFunction> coefficient)
{
  // Assign to all parts of form
  for (std::size_t part = 0; part < _form.num_parts(); part++)
     {
       Form& a = const_cast<Form&>(*_form.part(part));
       a.set_coefficient(_number, coefficient->part(part));
     }
}
//-----------------------------------------------------------------------------
