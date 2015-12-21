// Copyright (C) 2008 Anders Logg
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
// Modified by Garth N. Wells 2009
//
// First added:  2008-11-03
// Last changed: 2014-06-11

#include <dolfin/common/NoDeleter.h>
#include "SubSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SubSpace::SubSpace(const FunctionSpace& V, std::size_t component)
  : FunctionSpace(V.mesh(), V.element(), V.dofmap())
{
  // Extract subspace and assign
  std::shared_ptr<FunctionSpace>
    _function_space(V.extract_sub_space({component}));
  *static_cast<FunctionSpace*>(this) = *_function_space;
}
//-----------------------------------------------------------------------------
SubSpace::SubSpace(const FunctionSpace& V, std::size_t component,
                   std::size_t sub_component)
  : FunctionSpace(V.mesh(), V.element(), V.dofmap())
{
  // Extract subspace and assign
  std::shared_ptr<FunctionSpace>
    _function_space(V.extract_sub_space({component, sub_component}));
  *static_cast<FunctionSpace*>(this) = *_function_space;
}
//-----------------------------------------------------------------------------
SubSpace::SubSpace(const FunctionSpace& V,
                   const std::vector<std::size_t>& component)
  : FunctionSpace(V.mesh(), V.element(), V.dofmap())
{
  // Extract subspace and assign
  std::shared_ptr<FunctionSpace>
    _function_space(V.extract_sub_space(component));
  *static_cast<FunctionSpace*>(this) = *_function_space;
}
//-----------------------------------------------------------------------------
