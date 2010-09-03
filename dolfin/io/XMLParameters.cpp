// Copyright (C) 2004-2009 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-16
// Last changed: 2009-11-11

//#include <stdlib.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/parameter/Parameters.h>
#include "XMLIndent.h"
#include "XMLFile.h"
#include "XMLParameters.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLParameters::XMLParameters(Parameters& parameters, XMLFile& parser)
  : XMLHandler(parser), parameters(parameters), state(OUTSIDE)
{
  // Note that we don't clear the parameters here, only add new parameters
  // or overwrite existing parameters
}
//-----------------------------------------------------------------------------
void XMLParameters::start_element(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:

    if ( xmlStrcasecmp(name,(xmlChar *) "parameters") == 0 )
      state = INSIDE_PARAMETERS;

    break;

  case INSIDE_PARAMETERS:

    if ( xmlStrcasecmp(name,(xmlChar *) "parameter") == 0 )
      read_parameter(name,attrs);

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLParameters::end_element(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_PARAMETERS:

    if ( xmlStrcasecmp(name,(xmlChar *) "parameters") == 0 )
    {
      state = DONE;
      release();
    }

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLParameters::write(const Parameters& parameters,
                          std::ostream& outfile,
                          uint indentation_level)
{
  // FIXME: Handle nested parameter sets

  // Get keys
  std::vector<std::string> parameter_keys;
  parameters.get_parameter_keys(parameter_keys);

  // Write parameters header
  XMLIndent indent(indentation_level);
  outfile << indent() << "<parameters>" << std::endl;

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
    else if (parameter.type_str() == "real")
    {
      // FIXME: Cast to double here, extended precision lost
      outfile << "<parameter key=\"" << parameter.key() << "\" type=\"real\" value=\""
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

  // Write parameters footer
  outfile << indent() << "</parameters>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLParameters::read_parameter(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  const std::string key  = parse_string(name, attrs, "key");
  const std::string type = parse_string(name, attrs, "type");
  const std::string string_value = parse_string(name, attrs, "value");

  // Set parameter
  if (type == "double" || type == "real")
  {
    std::istringstream ss(string_value);
    double value;
    ss >> value;
    if (parameters.has_key(key))
      parameters[key] = value;
    else
      parameters.add(key, value);
  }
  else if (type == "int")
  {
    std::istringstream ss(string_value);
    int value;
    ss >> value;
    if (parameters.has_key(key))
      parameters[key] = value;
    else
      parameters.add(key, value);
  }
  else if (type == "bool")
  {
    bool value = true;
    if (string_value == "true")
      value = true;
    else if (string_value == "false" )
      value = false;
    else
      warning("Illegal value for boolean parameter: %s.", key.c_str());

    if (parameters.has_key(key))
      parameters[key] = value;
    else
      parameters.add(key, value);
  }
  else if (type == "string")
  {
    std::string value = string_value;
    if (parameters.has_key(key))
      parameters[key] = value;
    else
      parameters.add(key, value);
  }
  else
    warning("Illegal parameter type: %s", type.c_str());
}
//-----------------------------------------------------------------------------
