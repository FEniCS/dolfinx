// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdlib.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ParameterList.h>
#include <dolfin/XMLParameterList.h>

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
  switch ( state ) {
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
  switch ( state ) {
  case INSIDE_PARAMETERS:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "parameters") == 0 )
    {
      ok = true;
      state = DONE;
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLParameterList::reading(std::string filename)
{
  cout << "Loading parameters from file \"" << filename << "\"." << endl;
}
//-----------------------------------------------------------------------------
void XMLParameterList::done()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLParameterList::readParameter(const xmlChar *name, const xmlChar **attrs)
{
  // Parameter data
  std::string pname;
  std::string ptype;
  std::string pvalue;

  // Parse values
  parseStringRequired(name, attrs, "name",  pname);
  parseStringRequired(name, attrs, "type",  ptype);
  parseStringRequired(name, attrs, "value", pvalue);

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
      dolfin_warning1("Illegal value for boolean parameter: %s.", pname.c_str());
  }
  else if ( ptype == "string" )
  {
    parameters.set(pname.c_str(), pvalue.c_str());
  }
  else
    dolfin_warning1("Illegal parameter type: %s", ptype.c_str());
}
//-----------------------------------------------------------------------------
