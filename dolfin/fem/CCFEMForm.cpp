// Copyright (C) 2013 Anders Logg
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
// First added:  2013-09-12
// Last changed: 2013-09-12

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include "CCFEMForm.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CCFEMForm::CCFEMForm(boost::shared_ptr<const CCFEMFunctionSpace> function_space)
  : _function_space(function_space)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CCFEMForm::CCFEMForm(const CCFEMFunctionSpace& function_space)
  : _function_space(reference_to_no_delete_pointer(function_space))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CCFEMForm::~CCFEMForm()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void CCFEMForm::add(boost::shared_ptr<const Form> form)
{
  _forms.push_back(form);
  log(PROGRESS, "Added form to CCFEM form; form has %d part(s).",
      _forms.size());
}
//-----------------------------------------------------------------------------
void CCFEMForm::add(const Form& form)
{
  add(reference_to_no_delete_pointer(form));
}
//-----------------------------------------------------------------------------
void CCFEMForm::build()
{
  log(PROGRESS, "Building CCFEM form.");

  // FIXME: Not doing anything yet
}
//-----------------------------------------------------------------------------
