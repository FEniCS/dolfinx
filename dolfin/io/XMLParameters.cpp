// Copyright (C) 2004-2009 Ola Skavhaug and Anders Logg
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
// First added:  2009-03-16
// Last changed: 2011-03-28

#include "pugixml.hpp"
#include <dolfin/log/dolfin_log.h>
#include <dolfin/parameter/Parameters.h>
#include <dolfin/parameter/Parameter.h>
#include "XMLIndent.h"
#include "XMLParameters.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLParameters::read(Parameters& p, const pugi::xml_node xml_dolfin)
{
  // Check that we have a XML Parameters
  const pugi::xml_node xml_parameters = xml_dolfin.child("parameters");
  if (!xml_parameters)
    error("Not a DOLFIN Parameters file.");

  // FIXME: Nest parameters not yet supported
  // Check for nested parameters
  if (xml_dolfin.first_child().next_sibling())
    error("Reading of nested parameters from XML files is not yet supported.");

  // Get name of parameters ad rename paramter set
  const std::string name = xml_parameters.attribute("name").value();
  p.rename(name);

  // Iterate over parameters
  for (pugi::xml_node_iterator it = xml_parameters.begin(); it != xml_parameters.end(); ++it)
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
      error("Parameter type unknown in XMLParameters::read.");
  }
}
//-----------------------------------------------------------------------------
void XMLParameters::write(const Parameters& parameters,
                          std::ostream& outfile,
                          uint indentation_level)
{
  // Get keys
  std::vector<std::string> parameter_keys;
  parameters.get_parameter_keys(parameter_keys);

  // Write parameters header
  XMLIndent indent(indentation_level);
  outfile << indent() << "<parameters name=\"" << parameters.name() << "\">" << std::endl;

  // Write each parameter item
  ++indent;
  for (uint i = 0; i < parameter_keys.size(); ++i)
  {
    const Parameter& parameter = parameters[parameter_keys[i]];
    outfile << indent();
    if (parameter.type_str() == "int")
    {
      outfile << "<parameter key=\"" << parameter.key() << "\" type=\"int\" value=\""
              << static_cast<int>(parameter) << "\"/>" << std::endl;
    }
    else if (parameter.type_str() == "double")
    {
      outfile << "<parameter key=\"" << parameter.key() << "\" type=\"double\" value=\""
              << static_cast<double>(parameter) << "\"/>" << std::endl;
    }
    else if (parameter.type_str() == "bool")
    {
      if (static_cast<bool>(parameter))
        outfile << "<parameter key=\"" << parameter.key() << "\" type=\"bool\" value=\"true\"/>" << std::endl;
      else
        outfile << "<parameter key=\"" << parameter.key() << "\" type=\"bool\" value=\"false\"/>" << std::endl;
    }
    else if (parameter.type_str() == "string")
    {
      outfile << "<parameter key=\"" << parameter.key() << "\" type=\"string\" value=\""
              << static_cast<std::string>(parameter) << "\"/>" << std::endl;
    }
    else
    {
      error("Unable to write parameter \"%s\" to XML file, unknown type: \"%s\".",
            parameter.key().c_str(), parameter.type_str().c_str());
    }
  }
  --indent;

  // Write nested parameter sets
  std::vector<std::string> nested_keys;
  parameters.get_parameter_set_keys(nested_keys);
  for (uint i = 0; i < nested_keys.size(); ++i)
    write(parameters(nested_keys[i]), outfile, indentation_level + 1);

  // Write parameters footer
  outfile << indent() << "</parameters>" << std::endl;
}
//-----------------------------------------------------------------------------
template<class T>
void XMLParameters::add_parameter(Parameters& p, const std::string& key,
                                  T value)
{
  if (p.has_key(key))
    p[key] = value;
  else
    p.add(key, value);
}
//-----------------------------------------------------------------------------
