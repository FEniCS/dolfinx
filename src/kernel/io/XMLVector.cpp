// Copyright (C) 2002-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002-12-06
// Last changed: 2006-05-23


#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <dolfin/XMLVector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLVector::XMLVector(Vector& vector) : XMLObject(), x(vector)
{
  state = OUTSIDE;
}
//-----------------------------------------------------------------------------
void XMLVector::startElement(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vector") == 0 )
    {
      readVector(name, attrs);
      state = INSIDE_VECTOR;
    }
    
    break;
    
  case INSIDE_VECTOR:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "entry") == 0 )
      readEntry(name, attrs);
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLVector::endElement(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_VECTOR:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vector") == 0 )
    {
      state = DONE;
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLVector::readVector(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint size = parseInt(name, attrs, "size");
  
  // Initialize vector
  x.init(size);	 
}
//-----------------------------------------------------------------------------
void XMLVector::readEntry(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint row   = parseUnsignedInt(name, attrs, "row");
  real value = parseReal(name, attrs, "value");
  
  // Check values
  if ( row >= x.size() )
    dolfin_error2("Illegal XML data for Vector: row index %d out of range (0 - %d)",
		  row, x.size() - 1);
  
  // Set value
  x(row) = value;
}
//-----------------------------------------------------------------------------
