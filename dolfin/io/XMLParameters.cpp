// Copyright (C) 2011 Garth N. Wells
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
// Modified by Anders Logg 2011
//
// First added:  2009-03-16
// Last changed: 2011-11-14

#include "pugixml.hpp"
#include <dolfin/log/log.h>
#include <dolfin/parameter/Parameter.h>
#include <dolfin/parameter/Parameters.h>
#include "XMLParameters.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLParameters::read(Parameters& p, const pugi::xml_node xml_dolfin)
{
  // Check that we have a XML Parameters
  const pugi::xml_node xml_parameters = xml_dolfin.child("parameters");
  if (!xml_parameters)
  {
    dolfin_error("XMLParameters.cpp",
                 "read parameters from XML file",
                 "Not a DOLFIN Parameters XML file");
  }

  // Check that there is only one root parameters set
  if (xml_dolfin.first_child().next_sibling())
  {
    dolfin_error("XMLParameters.cpp",
                 "read parameters from XML file",
                 "Two parameter sets (not nested) are defined in XML file");
  }

  // Get name of root parameters and rename parameter set
  const std::string name = xml_parameters.attribute("name").value();
  p.rename(name);

  // Read parameters
  read_parameter_nest(p, xml_parameters);
}
//-----------------------------------------------------------------------------
void XMLParameters::write(const Parameters& parameters, pugi::xml_node xml_node)
{
  // Get keys
  std::vector<std::string> parameter_keys;
  parameters.get_parameter_keys(parameter_keys);

  // Add parameters node
  pugi::xml_node parameters_node = xml_node.append_child("parameters");
  parameters_node.append_attribute("name") = parameters.name().c_str();

  // Write each parameter item
  for (std::size_t i = 0; i < parameter_keys.size(); ++i)
  {
    // Get parameter
    const Parameter& parameter = parameters[parameter_keys[i]];

    // Add parameter, if set
    if (parameter.is_set())
    {
      pugi::xml_node parameter_node = parameters_node.append_child("parameter");
      parameter_node.append_attribute("key") = parameter.key().c_str();
      parameter_node.append_attribute("type") = parameter.type_str().c_str();
      if (parameter.type_str() == "int")
        parameter_node.append_attribute("value") = static_cast<int>(parameter);
      else if (parameter.type_str() == "double")
      {
        parameter_node.append_attribute("value")
          = static_cast<double>(parameter);
      }
      else if (parameter.type_str() == "bool")
        parameter_node.append_attribute("value") = static_cast<bool>(parameter);
      else if (parameter.type_str() == "string")
      {
        parameter_node.append_attribute("value")
          = static_cast<std::string>(parameter).c_str();
      }
      else
      {
        dolfin_error("XMLParameters.cpp",
                     "write parameters to XML file",
                     "Unknown type (\"%s\") of parameters \"%s\"",
                     parameter.type_str().c_str(), parameter.key().c_str());
      }
    }
  }

  // Write nested parameter sets
  std::vector<std::string> nested_keys;
  parameters.get_parameter_set_keys(nested_keys);
  for (std::size_t i = 0; i < nested_keys.size(); ++i)
    write(parameters(nested_keys[i]), parameters_node);
}
//-----------------------------------------------------------------------------
void XMLParameters::read_parameter_nest(Parameters& p,
                                        const pugi::xml_node xml_node)
{
  // Iterate over parameters
  for (auto it = xml_node.begin(); it != xml_node.end(); ++it)
  {
    // Get name (parameters or parameter)
    const std::string node_name = it->name();
    if (node_name == "parameters")
    {
      // Get name of parameters set
      const std::string name = it->attribute("name").value();

      // Create parameter set if necessary
      if (!p.has_parameter_set(name))
      {
        Parameters nested_parameters(name);
        p.add(nested_parameters);
      }

      // Set parameter value
      read_parameter_nest(p(name), *it);
    }
    else if (node_name == "parameter")
    {
      const std::string key = it->attribute("key").value();
      const std::string type = it->attribute("type").value();
      const pugi::xml_attribute value = it->attribute("value");
      if (type == "double")
        XMLParameters::add_parameter(p, key, value.as_double());
      else if (type == "int")
        XMLParameters::add_parameter(p, key, value.as_int());
      else if (type == "bool")
        XMLParameters::add_parameter(p, key, value.as_bool());
      else if (type == "string")
        XMLParameters::add_parameter(p, key, value.value());
      else
      {
        dolfin_error("XMLParameters.cpp",
                     "read parameters from XML file",
                     "Unknown type (\"%s\") of parameters \"%s\"",
                     type.c_str(), key.c_str());
      }
    }
    else
    {
      dolfin_error("XMLParameters.cpp",
                   "read parameters from XML file",
                   "Unknown tag (\"%s\") in XML Parameters file",
                   node_name.c_str());
    }
  }
}
//-----------------------------------------------------------------------------
template<typename T>
void XMLParameters::add_parameter(Parameters& p, const std::string& key,
                                  T value)
{
  if (p.has_parameter(key))
    p[key] = value;
  else
    p.add(key, value);
}
//-----------------------------------------------------------------------------
