// Copyright (C) 2012 Fredrik Valdmanis
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
// Modified by Joachim B Haga 2012
//
// First added:  2012-06-20
// Last changed: 2012-09-20

#ifdef HAS_VTK

#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericVector.h>

#include "VTKPlottableDirichletBC.h"

using namespace dolfin;

//----------------------------------------------------------------------------
VTKPlottableDirichletBC::VTKPlottableDirichletBC(std::shared_ptr<const DirichletBC> bc)
  : VTKPlottableGenericFunction(std::shared_ptr<const Function>(new Function(bc->function_space()))),
    _bc(bc)
{
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKPlottableDirichletBC::init_pipeline(const Parameters& parameters)
{
  Parameters new_parameters = parameters;
  new_parameters["mode"] = "color";
  VTKPlottableGenericFunction::init_pipeline(new_parameters);
}
//----------------------------------------------------------------------------
bool VTKPlottableDirichletBC::is_compatible(const Variable &var) const
{
  const DirichletBC *bc(dynamic_cast<const DirichletBC*>(&var));
  if (!bc)
  {
    return false;
  }

  const FunctionSpace &V = *bc->function_space();
  if (V.element()->value_rank() > 1 || V.element()->value_rank() != _bc.lock()->function_space()->element()->value_rank())
  {
    return false;
  }

  return VTKPlottableMesh::is_compatible(*V.mesh());
}
//----------------------------------------------------------------------------
void VTKPlottableDirichletBC::update(std::shared_ptr<const Variable> var,
                                     const Parameters& parameters,
                                     int framecounter)
{
  if (var)
    _bc = std::dynamic_pointer_cast<const DirichletBC>(var);

  dolfin_assert(!_function.expired());
  std::shared_ptr<const Function> func
    = std::dynamic_pointer_cast<const Function>(_function.lock());

  auto bc = _bc.lock();

  dolfin_assert(bc && func);
  if (bc->function_space() != func->function_space())
    func.reset(new Function(bc->function_space()));

  // We passed in the Function to begin with, so the const_case is safe
  GenericVector &vec = *const_cast<GenericVector*>(func->vector().get());

  double unset_value = 0.0;
  if (func->value_rank() == 0)
    unset_value = std::numeric_limits<double>::quiet_NaN();

  // Set the function data to all-undefined (zero for vectors, NaN for scalars)
  std::vector<double> data(vec.local_size());
  std::fill(data.begin(), data.end(), unset_value);
  vec.set_local(data);

  // Apply the BC
  bc->apply(vec);

  VTKPlottableGenericFunction::update(func, parameters, framecounter);
}
//----------------------------------------------------------------------------
VTKPlottableDirichletBC *dolfin::CreateVTKPlottable(std::shared_ptr<const DirichletBC> bc)
{
  return new VTKPlottableDirichletBC(bc);
}

#endif
