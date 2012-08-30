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
// Last changed: 2012-08-30

#ifdef HAS_VTK

#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericVector.h>

#include "VTKPlottableDirichletBC.h"

using namespace dolfin;

//----------------------------------------------------------------------------
VTKPlottableDirichletBC::VTKPlottableDirichletBC(boost::shared_ptr<const DirichletBC> bc) :
  _bc(bc),
  VTKPlottableGenericFunction(boost::shared_ptr<const Function>(new Function(bc->function_space())))
{
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKPlottableDirichletBC::init_pipeline(const Parameters& parameters)
{
  Parameters new_parameters = parameters;
  new_parameters["mode"] = "off";
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
  if (V.element()->value_rank() > 1 || V.element()->value_rank() != _bc->function_space()->element()->value_rank())
  {
    return false;
  }

  return VTKPlottableMesh::is_compatible(*V.mesh());
}
//----------------------------------------------------------------------------
void VTKPlottableDirichletBC::update(boost::shared_ptr<const Variable> var, const Parameters& parameters, int framecounter)
{
  if (var)
  {
    _bc = boost::dynamic_pointer_cast<const DirichletBC>(var);
  }

  boost::shared_ptr<const Function> func = boost::dynamic_pointer_cast<const Function>(_function);

  dolfin_assert(_bc && func);

  if (_bc->function_space() != func->function_space())
  {
    func.reset(new Function(_bc->function_space()));
  }

  // We passed in the Function to begin with, so the const_case is safe
  GenericVector &vec = *const_cast<GenericVector*>(func->vector().get());

  double unset_value = 0.0;
  if (func->value_rank() == 0)
  {
    unset_value = std::numeric_limits<double>::quiet_NaN();
  }

  // Set the function data to all-undefined (zero for vectors, NaN for scalars)
  std::vector<double> data(vec.local_size());
  std::fill(data.begin(), data.end(), unset_value);
  vec.set_local(data);

  // Apply the BC
  _bc->apply(vec);

  VTKPlottableGenericFunction::update(func, parameters, framecounter);
}
//----------------------------------------------------------------------------

#endif
