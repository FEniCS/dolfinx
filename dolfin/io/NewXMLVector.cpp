// Copyright (C) 2002-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2002-12-06
// Last changed: 2006-05-23


#include <dolfin/log/dolfin_log.h>
#include <dolfin/la/Vector.h>
#include "NewXMLVector.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NewXMLVector::NewXMLVector(GenericVector& vector, NewXMLFile& parser, bool inside)
  : XMLHandler(parser), x(vector), state(OUTSIDE), values(0)
{
  if (inside)
    state = INSIDE_VECTOR;
}
//-----------------------------------------------------------------------------
void NewXMLVector::start_element(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vector") == 0 )
    {
      start_vector(name, attrs);
      state = INSIDE_VECTOR;
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void NewXMLVector::end_element(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_VECTOR:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vector") == 0 )
    {
      end_vector();
      state = DONE;
      release();
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void NewXMLVector::start_vector(const xmlChar *name, const xmlChar **attrs)
{
  dolfin_assert(values == 0);
  values = new std::vector<double>();
  xml_array = new XMLArray(*values, parser);
  xml_array->handle();
}
//-----------------------------------------------------------------------------
void NewXMLVector::end_vector()
{
  // Copy values to vector
  x.resize(values->size());
  double v[values->size()];
  for (uint i = 0; i< values->size(); ++i)
    v[i] = (*values)[i];
  x.set(v);
  delete values;
  delete xml_array;
  xml_array = 0;
}
//-----------------------------------------------------------------------------
