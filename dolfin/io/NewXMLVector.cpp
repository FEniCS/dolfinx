// Copyright (C) 2002-2006 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-06
// Last changed: 2009-03-16


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
void NewXMLVector::write(const GenericVector& vector, std::ostream& outfile, uint indentation_level)
{
  // Write vector header
  uint curr_indent = indentation_level;
  outfile << std::setw(curr_indent) << "";
  outfile << "<vector>" << std::endl;

  uint size = vector.size();
  double vector_values[size];
  vector.get(vector_values);

  // Convert Vector values to std::vector<double>
  std::vector<double> arr;
  arr.resize(size);
  for (uint i = 0; i < size; ++i)
    arr[i] = vector_values[i];
  // Write array
  XMLArray::write(arr, outfile, indentation_level + 2);

  // Write vector footer
  curr_indent = indentation_level;
  outfile << std::setw(curr_indent) << "";
  outfile << "</vector>" << std::endl;

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
