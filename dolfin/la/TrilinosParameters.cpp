// Copyright (C) 2015 Chris Richardson
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

#ifdef HAS_TRILINOS

#include "TrilinosParameters.h"

using namespace dolfin;
//-----------------------------------------------------------------------------
void TrilinosParameters::insert_parameters(const Parameters& params,
                  Teuchos::RCP<Teuchos::ParameterList> parameter_list)
{
  // Get keys of Parameter in params
  std::vector<std::string> keys;
  params.get_parameter_keys(keys);

  for (auto &k : keys)
  {
    // Replace "_" with " " in parameter key
    // because dolfin parameters cannot have spaces in the key name
    std::string trilinos_key = k;
    for (auto &c : trilinos_key)
      if (c == '_')
        c = ' ';

    const std::string type = params[k].type_str();

    if (type == "int")
      parameter_list->set(trilinos_key, int(params[k]));
    else if (type == "double")
      parameter_list->set(trilinos_key, double(params[k]));
    else if (type == "bool")
      parameter_list->set(trilinos_key, bool(params[k]));
    else if (type == "string")
      parameter_list->set(trilinos_key, std::string(params[k]));
    else
    {
      dolfin_error("TrilinosParameters.cpp",
                   "set parameter",
                   "Cannot parse type \"%s\"", type.c_str());
    }
  }

  // Get keys of any parameter subsets in the dolfin parameters
  keys.clear();
  params.get_parameter_set_keys(keys);
  for (auto &k : keys)
  {
    // Replace "_" with " " in parameter key
    // because dolfin parameters cannot have spaces in the key name
    std::string trilinos_key = k;
    for (auto &c : trilinos_key)
      if (c == '_')
        c = ' ';

    // Create a new ParameterList
    Teuchos::RCP<Teuchos::ParameterList>
      sub_parameter_list(new Teuchos::ParameterList(trilinos_key));
    // Recursively call to add parameters
    insert_parameters(params(k), sub_parameter_list);

    parameter_list->set(trilinos_key, *sub_parameter_list);

  }
}
//-----------------------------------------------------------------------------
#endif
