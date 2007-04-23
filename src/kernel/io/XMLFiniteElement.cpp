// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-13
// Last changed: 2007-04-13

#include <dolfin/dolfin_log.h>
#include <dolfin/XMLFiniteElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLFiniteElement::XMLFiniteElement(std::string& signature)
  : XMLObject(), signature(signature)
{
  state = OUTSIDE;
}
//-----------------------------------------------------------------------------
void XMLFiniteElement::startElement(const xmlChar* name, const xmlChar** attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "finiteelement") == 0 )
    {
      readFiniteElement(name, attrs);
      state = INSIDE_FINITE_ELEMENT;
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLFiniteElement::endElement(const xmlChar* name)
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
void XMLFiniteElement::readFiniteElement(const xmlChar* name,
                                         const xmlChar** attrs)
{
  // Parse values
  signature = parseString(name, attrs, "signature");
}
//-----------------------------------------------------------------------------
