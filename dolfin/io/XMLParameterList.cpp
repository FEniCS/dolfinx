// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2004-03-31
// Last changed: 2006-05-23

#include <stdlib.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/parameter/ParameterList.h>
#include "XMLParameterList.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLParameterList::XMLParameterList(ParameterList& parameters)
  : XMLObject(), parameters(parameters)
{
  state = OUTSIDE;
}
//-----------------------------------------------------------------------------
void XMLParameterList::startElement(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "parameters") == 0 )
      state = INSIDE_PARAMETERS;
    
    break;

  case INSIDE_PARAMETERS:

    if ( xmlStrcasecmp(name,(xmlChar *) "parameter") == 0 )
      readParameter(name,attrs);
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLParameterList::endElement(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_PARAMETERS:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "parameters") == 0 )
    {
      state = DONE;
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLParameterList::open(std::string filename)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool XMLParameterList::close()
{
  return state == DONE;
}
//-----------------------------------------------------------------------------
void XMLParameterList::readParameter(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  std::string pname  = parseString(name, attrs, "name");
  std::string ptype  = parseString(name, attrs, "type");
  std::string pvalue = parseString(name, attrs, "value");

  // Set parameter
  if ( ptype == "real" )
  {
    real val = atof(pvalue.c_str());
    parameters.set(pname.c_str(), val);
  }
  else if ( ptype == "int" )
  {
    int val = atoi(pvalue.c_str());
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
