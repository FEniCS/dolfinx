// Copyright (C) 2013-2015 Anders Logg
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
// Last changed: 2015-11-05

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/MultiMeshFunctionSpace.h>
#include "Form.h"
#include "MultiMeshForm.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMeshForm::MultiMeshForm(std::shared_ptr<const MultiMesh> multimesh)
  : _rank(0)
{
  _multimesh = multimesh;
}
//-----------------------------------------------------------------------------
MultiMeshForm::MultiMeshForm(const MultiMesh& multimesh)
  : _rank(0)
{
  _multimesh = reference_to_no_delete_pointer(multimesh);
}
//-----------------------------------------------------------------------------
MultiMeshForm::MultiMeshForm(std::shared_ptr<const MultiMeshFunctionSpace> function_space)
  : _rank(0)
{
  _function_spaces.push_back(function_space);
}
//-----------------------------------------------------------------------------
MultiMeshForm::MultiMeshForm(const MultiMeshFunctionSpace& function_space)
  : _rank(0)
{
  _function_spaces.push_back(reference_to_no_delete_pointer(function_space));
}
//-----------------------------------------------------------------------------
MultiMeshForm::MultiMeshForm(std::shared_ptr<const MultiMeshFunctionSpace> function_space_0,
                     std::shared_ptr<const MultiMeshFunctionSpace> function_space_1)
  : _rank(0)
{
  _function_spaces.push_back(function_space_0);
  _function_spaces.push_back(function_space_1);
}
//-----------------------------------------------------------------------------
MultiMeshForm::MultiMeshForm(const MultiMeshFunctionSpace& function_space_0,
                     const MultiMeshFunctionSpace& function_space_1)
  : _rank(0)
{
  _function_spaces.push_back(reference_to_no_delete_pointer(function_space_0));
  _function_spaces.push_back(reference_to_no_delete_pointer(function_space_1));
}
//-----------------------------------------------------------------------------
MultiMeshForm::~MultiMeshForm()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t MultiMeshForm::rank() const
{
  return _rank;
}
//-----------------------------------------------------------------------------
std::size_t MultiMeshForm::num_parts() const
{
  return _forms.size();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const MultiMesh> MultiMeshForm::multimesh() const
{
  // Return multimesh if stored (from functional)
  if (_multimesh)
    return _multimesh;

  // Extract meshes from function spaces
  std::vector<std::shared_ptr<const MultiMesh>> multimeshes;
  for (std::size_t i = 0; i < _function_spaces.size(); i++)
  {
    dolfin_assert(_function_spaces[i]->multimesh());
    multimeshes.push_back(_function_spaces[i]->multimesh());
  }

  // Check that we have at least one multimesh
  if (multimeshes.empty())
  {
    dolfin_error("MultiMeshForm.cpp",
                 "extract multimesh from form",
                 "No multimesh was found");
  }

  // Check that all multimeshes are the same
  for (std::size_t i = 1; i < multimeshes.size(); i++)
  {
    if (multimeshes[i] != multimeshes[i - 1])
    {
      dolfin_error("MultiMeshForm.cpp",
                   "extract multimesh from form",
                   "Non-matching multimeshes for function spaces");
    }
  }

  // Return first multimesh
  dolfin_assert(multimeshes[0]);
  return multimeshes[0];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Form> MultiMeshForm::part(std::size_t i) const
{
  dolfin_assert(i < _forms.size());
  return _forms[i];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const MultiMeshFunctionSpace>
MultiMeshForm::function_space(std::size_t i) const
{
  dolfin_assert(i < _function_spaces.size());
  return _function_spaces[i];
}
//-----------------------------------------------------------------------------
void MultiMeshForm::add(std::shared_ptr<const Form> form)
{
  _forms.push_back(form);
  log(PROGRESS, "Added form to MultiMesh form; form has %d part(s).",
      _forms.size());
}
//-----------------------------------------------------------------------------
void MultiMeshForm::add(const Form& form)
{
  add(reference_to_no_delete_pointer(form));
}
//-----------------------------------------------------------------------------
void MultiMeshForm::build()
{
  log(PROGRESS, "Building MultiMesh form.");

  // Check rank
  for (std::size_t i = 1; i < num_parts(); i++)
  {
    std::size_t r0 = _forms[i - 1]->rank();
    std::size_t r1 = _forms[i]->rank();
    if (r0 != r1)
    {
      dolfin_error("MultiMeshForm.cpp",
                   "build MultiMesh form",
                   "Non-matching ranks (%d and %d) for forms.", r0, r1);
    }
  }

  // Set rank
  _rank = _forms[0]->rank();
}
//-----------------------------------------------------------------------------
void MultiMeshForm::clear()
{
  _rank = 0;
  _multimesh = std::shared_ptr<MultiMesh>();
  _function_spaces.clear();
  _forms.clear();
}
//-----------------------------------------------------------------------------
