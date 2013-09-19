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
// GNU Lesser General Public License for more details.~1e-20.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2013-09-12
// Last changed: 2013-09-19

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include "Form.h"
#include "CCFEMForm.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CCFEMForm::CCFEMForm(boost::shared_ptr<const CCFEMFunctionSpace> function_space)
  : _rank(0)
{
  _function_spaces.push_back(function_space);
}
//-----------------------------------------------------------------------------
CCFEMForm::CCFEMForm(const CCFEMFunctionSpace& function_space)
  : _rank(0)
{
  _function_spaces.push_back(reference_to_no_delete_pointer(function_space));
}
//-----------------------------------------------------------------------------
CCFEMForm::CCFEMForm(boost::shared_ptr<const CCFEMFunctionSpace> function_space_0,
                     boost::shared_ptr<const CCFEMFunctionSpace> function_space_1)
  : _rank(0)
{
  _function_spaces.push_back(function_space_0);
  _function_spaces.push_back(function_space_1);
}
//-----------------------------------------------------------------------------
CCFEMForm::CCFEMForm(const CCFEMFunctionSpace& function_space_0,
                     const CCFEMFunctionSpace& function_space_1)
  : _rank(0)
{
  _function_spaces.push_back(reference_to_no_delete_pointer(function_space_0));
  _function_spaces.push_back(reference_to_no_delete_pointer(function_space_1));
}
//-----------------------------------------------------------------------------
CCFEMForm::~CCFEMForm()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t CCFEMForm::rank() const
{
  return _rank;
}
//-----------------------------------------------------------------------------
std::size_t CCFEMForm::num_parts() const
{
  return _forms.size();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const Form> CCFEMForm::part(std::size_t i) const
{
  dolfin_assert(i < _forms.size());
  return _forms[i];
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const CCFEMFunctionSpace>
CCFEMForm::function_space(std::size_t i) const
{
  dolfin_assert(i < _function_spaces.size());
  return _function_spaces[i];
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

  // Check rank
  for (std::size_t i = 1; i < num_parts(); i++)
  {
    std::size_t r0 = _forms[i - 1]->rank();
    std::size_t r1 = _forms[i]->rank();
    if (r0 != r1)
    {
      dolfin_error("CCFEMForm.cpp",
                   "build CCFEM form",
                   "Non-matching ranks (%d and %d) for forms.", r0, r1);
    }
  }

  // Set rank
  _rank = _forms[0]->rank();
}
//-----------------------------------------------------------------------------
void CCFEMForm::clear()
{
  _rank = 0;
  _function_spaces.clear();
  _forms.clear();
}
//-----------------------------------------------------------------------------
