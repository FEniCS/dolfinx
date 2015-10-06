// Copyright (C) 2013 Johan Hake
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
// First added:  2013-11-07
// Last changed: 2013-11-07

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include "assign.h"
#include "FunctionAssigner.h"

//-----------------------------------------------------------------------------
void dolfin::assign(std::shared_ptr<Function> receiving_func,
		    std::shared_ptr<const Function> assigning_func)
{
  // Instantiate FunctionAssigner and call assign
  const FunctionAssigner assigner(receiving_func->function_space(),
				  assigning_func->function_space());
  assigner.assign(receiving_func, assigning_func);
}
//-----------------------------------------------------------------------------
void
dolfin::assign(std::shared_ptr<Function> receiving_func,
               std::vector<std::shared_ptr<const Function>> assigning_funcs)
{

  // Instantiate FunctionAssigner and call assign
  std::vector<std::shared_ptr<const FunctionSpace>> assigning_spaces;
  for (std::size_t i = 0; i < assigning_funcs.size(); i++)
    assigning_spaces.push_back(assigning_funcs[i]->function_space());

  const FunctionAssigner assigner(receiving_func->function_space(),
				  assigning_spaces);
  assigner.assign(receiving_func, assigning_funcs);

}
//-----------------------------------------------------------------------------
void dolfin::assign(std::vector<std::shared_ptr<Function>> receiving_funcs,
		    std::shared_ptr<const Function> assigning_func)
{
  // Instantiate FunctionAssigner and call assign
  std::vector<std::shared_ptr<const FunctionSpace>> receiving_spaces;

  for (std::size_t i = 0; i < receiving_funcs.size(); i++)
    receiving_spaces.push_back(receiving_funcs[i]->function_space());

  const FunctionAssigner assigner(receiving_spaces,
				  assigning_func->function_space());
  assigner.assign(receiving_funcs, assigning_func);
}
//-----------------------------------------------------------------------------
