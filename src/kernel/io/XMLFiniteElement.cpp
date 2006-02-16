// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-16
// Last changed: 2006-02-16

#include <dolfin/dolfin_log.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/XMLFiniteElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLFiniteElement::XMLFiniteElement(FiniteElement& element)
  : XMLObject(), element(element)
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
      ok = true;
      state = DONE;
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLFiniteElement::readFiniteElement(const xmlChar* name, const xmlChar** attrs)
{
  // Set default values
  std::string type;
  std::string shape;
  int degree;
  int vectordim;

  // Parse values
  parseStringRequired (name, attrs, "type",      type);
  parseStringRequired (name, attrs, "shape",     shape);
  parseIntegerRequired(name, attrs, "degree",    degree);
  parseIntegerRequired(name, attrs, "vectordim", vectordim);
  
  // Don't know what to do here, maybe we need to make an envelope-letter
  // interface for finite element as well?
}
//-----------------------------------------------------------------------------
