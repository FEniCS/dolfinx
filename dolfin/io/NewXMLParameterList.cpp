// Copyright (C) 2004-2006 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-16
// Last changed: 2009-03-17

#include <stdlib.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/parameter/ParameterList.h>
#include "XMLIndent.h"
#include "NewXMLFile.h"
#include "NewXMLParameterList.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NewXMLParameterList::NewXMLParameterList(ParameterList& parameters, NewXMLFile& parser)
  : XMLHandler(parser), parameters(parameters), state(OUTSIDE)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewXMLParameterList::start_element(const xmlChar *name, const xmlChar **attrs)
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
void NewXMLParameterList::end_element(const xmlChar *name)
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
void NewXMLParameterList::write(const ParameterList& parameters, std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);

  // Write parameters header
  outfile << indent() << "<parameters>" << std::endl;

  // Write each parameter item
  ++indent;
  for (ParameterList::const_iterator it = parameters.parameters.begin(); it != parameters.parameters.end(); ++it)
  {
    const Parameter parameter = it->second;
    outfile << indent();
    switch ( parameter.type() )
    {
    case Parameter::type_int:
      outfile << "<parameter name=\"" << it->first << "\" type=\"int\" value=\"" << static_cast<int>(parameter) << "\"/>" << std::endl;
      break;
    case Parameter::type_real:
      outfile << "<parameter name=\"" << it->first << "\" type=\"real\" value=\"" << static_cast<double>(parameter) << "\"/>" << std::endl;
      break;
    case Parameter::type_bool:
      if (static_cast<bool>(parameter))
        outfile << "<parameter name=\"" << it->first << "\" type=\"bool\" value=\"true\"/>" << std::endl;
      else
        outfile << "<parameter name=\"" << it->first << "\" type=\"bool\" value=\"false\"/>" << std::endl;
      break;
    case Parameter::type_string:
      outfile << "<parameter name=\"" << it->first << "\" type=\"string\" value=\"" << static_cast<std::string>(parameter) << "\"/>" << std::endl;
      break;
    default:
      ; // Do nothing
    }
  }
  --indent;

  // Write parameters footer
  outfile << indent() << "</parameters>" << std::endl;
}
//-----------------------------------------------------------------------------
void NewXMLParameterList::read_parameter(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  std::string pname  = parse_string(name, attrs, "name");
  std::string ptype  = parse_string(name, attrs, "type");
  std::string pvalue = parse_string(name, attrs, "value");

  // Set parameter
  if ( ptype == "real" )
  {
    std::istringstream ss(pvalue);
    double val;
    ss >> val;
    parameters.set(pname.c_str(), val);
  }
  else if ( ptype == "int" )
  {
    std::istringstream ss(pvalue);
    int val;
    ss >> val;
    parameters.set(pname.c_str(), val);
  }
  else if ( ptype == "bool" )
  {
    if ( pvalue == "true" )
      parameters.set(pname.c_str(), true);
    else if ( pvalue == "false" )
      parameters.set(pname.c_str(), false);
    else
      warning("Illegal value for boolean parameter: %s.", pname.c_str());
  }
  else if ( ptype == "string" )
  {
    parameters.set(pname.c_str(), pvalue.c_str());
  }
  else
    warning("Illegal parameter type: %s", ptype.c_str());
}
//-----------------------------------------------------------------------------
