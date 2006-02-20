// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-19
// Last changed: 2006-02-20

#include <dolfin/dolfin_log.h>
#include <dolfin/FiniteElementSpec.h>
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
      ok = true;
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
  
  // Check data
  if ( degree < 0 )
    dolfin_error1("Illegal degree (%d) for finite element.", degree);
  if ( vectordim < 0 )
    dolfin_error1("Illegal vector dimension (%d) for finite element.", vectordim);
  uint degree_uint = static_cast<uint>(degree);
  uint vectordim_uint = static_cast<uint>(vectordim);

  // Don't know what to do here, maybe we need to make an envelope-letter
  spec.init(type, shape, degree_uint, vectordim_uint);
}
//-----------------------------------------------------------------------------
