// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-19
// Last changed: 2006-05-23

#include <dolfin/dolfin_log.h>
#include <dolfin/XMLFiniteElementSpec.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLFiniteElementSpec::XMLFiniteElementSpec(FiniteElementSpec& spec)
  : XMLObject(), spec(spec)
{
  state = OUTSIDE;
}
//-----------------------------------------------------------------------------
void XMLFiniteElementSpec::startElement(const xmlChar* name, const xmlChar** attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "finiteelement") == 0 )
    {
      readFiniteElementSpec(name, attrs);
      state = INSIDE_FINITE_ELEMENT;
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLFiniteElementSpec::endElement(const xmlChar* name)
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
void XMLFiniteElementSpec::readFiniteElementSpec(const xmlChar* name,
						 const xmlChar** attrs)
{
  dolfin_error("Reading finite element spec in XML format not implemented for new UFC structure.");
/*
  // Parse values
  std::string type  = parseString(name, attrs, "type");
  std::string shape = parseString(name, attrs, "shape");
  uint degree       = parseUnsignedInt(name, attrs, "degree");
  uint vectordim    = parseUnsignedInt(name, attrs, "vectordim");
  
  // FIXME: Why is this comment here?

  // Don't know what to do here, maybe we need to make an envelope-letter
  spec.init(type, shape, degree, vectordim);
*/
}
//-----------------------------------------------------------------------------
