// Copyright (C) 2002-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002-12-06
// Last changed: 2005

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
  switch ( state ) {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "vector") == 0 ) {
      readVector(name,attrs);
      state = INSIDE_VECTOR;
    }
    
    break;
    
  case INSIDE_VECTOR:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "element") == 0 )
      readElement(name,attrs);
    
    break;
    
  default:
    ;
  }
  
}
//-----------------------------------------------------------------------------
void XMLVector::endElement(const xmlChar *name)
{
  switch ( state ) {
  case INSIDE_VECTOR:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "vector") == 0 ) {
      ok = true;
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
  // Set default values
  int size = 0;

  // Parse values
  parseIntegerRequired(name, attrs, "size", size);
  
  // Check values
  if ( size < 0 )
    dolfin_error("Illegal XML data for Vector: size of vector must be positive.");
  
  // Initialise
  x.init(size);	 
}
//-----------------------------------------------------------------------------
void XMLVector::readElement(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int row = 0;
  real value = 0.0;
  
  // Parse values
  parseIntegerRequired(name, attrs, "row", row);
  parseRealRequired(name, attrs, "value",  value);   
  
  // Check values
  if ( row < 0 || row >= static_cast<int>(x.size()) )
    dolfin_error2("Illegal XML data for Vector: row index %d out of range (0 - %d)", row, x.size() - 1);
  
  // Set value
  x(row) = value;
}
//-----------------------------------------------------------------------------
