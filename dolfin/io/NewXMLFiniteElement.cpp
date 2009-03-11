// Copyright (C) 2007 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-11
// Last changed: 2009-03-11

#include <dolfin/log/dolfin_log.h>
#include "NewXMLFiniteElement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NewXMLFiniteElement::NewXMLFiniteElement(std::string& signature, NewXMLFile& parser)
  : XMLHandler(parser), signature(signature), state(OUTSIDE)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewXMLFiniteElement::start_element(const xmlChar* name, const xmlChar** attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "finteelement") == 0 )
    {
      read_finite_element(name, attrs);
      state = INSIDE_FINITE_ELEMENT;
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void NewXMLFiniteElement::end_element(const xmlChar* name)
{
  switch ( state )
  {
  case INSIDE_FINITE_ELEMENT:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "finiteelement") == 0 )
    {
      state = DONE;
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void NewXMLFiniteElement::write(const std::string& signature, std::ostream& outfile, uint indentation_level)
{
  outfile << std::setw(indentation_level) << "" << "<finiteelement signature=\"" << signature << "\"/>" << std::endl;
}
//-----------------------------------------------------------------------------
void NewXMLFiniteElement::read_finite_element(const xmlChar* name, const xmlChar** attrs)
{
  // Parse values
  signature = parse_string(name, attrs, "signature");
}
//-----------------------------------------------------------------------------
