// Copyright (C) 2009 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-02
// Last changed: 2009-03-17

#include <dolfin/common/Array.h>
#include <dolfin/log/dolfin_log.h>
#include "XMLFile.h"
#include "XMLIndent.h"
#include "XMLArray.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLArray::XMLArray(std::vector<int>& ix, XMLFile& parser)
  : XMLHandler(parser), ix(&ix), ux(0), dx(0),
    state(OUTSIDE_ARRAY), atype(INT)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLArray::XMLArray(std::vector<uint>& ux, XMLFile& parser)
  : XMLHandler(parser), ix(0), ux(&ux), dx(0), state(OUTSIDE_ARRAY),
    atype(UINT)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLArray::XMLArray(std::vector<double>& dx, XMLFile& parser)
  : XMLHandler(parser), ix(0), ux(0), dx(&dx), state(OUTSIDE_ARRAY),
    atype(DOUBLE)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLArray::XMLArray(std::vector<int>& ix, XMLFile& parser, uint size)
  : XMLHandler(parser), ix(&ix), ux(0), dx(0), state(INSIDE_ARRAY), atype(INT)
{
  indices.reserve(size);
  this->ix->clear();
  this->ix->reserve(size);
  std::fill(this->ix->begin(), this->ix->end(), 0);
}
//-----------------------------------------------------------------------------
XMLArray::XMLArray(std::vector<uint>& ux, XMLFile& parser, uint size)
  : XMLHandler(parser), ix(0), ux(&ux), dx(0), state(INSIDE_ARRAY), atype(UINT)
{
  indices.reserve(size);
  this->ux->clear();
  this->ux->reserve(size);
  std::fill(this->ux->begin(), this->ux->end(), 0);
}
//-----------------------------------------------------------------------------
XMLArray::XMLArray(std::vector<double>& dx,
                   XMLFile& parser, uint size)
  : XMLHandler(parser), indices(indices), ix(0), ux(0), dx(&dx), state(INSIDE_ARRAY),
    atype(DOUBLE)
{
  indices.reserve(size);
  this->dx->clear();
  this->dx->reserve(size);
  std::fill(this->dx->begin(), this->dx->end(), 0.0);
}
//-----------------------------------------------------------------------------
void XMLArray::start_element(const xmlChar* name, const xmlChar** attrs)
{
  switch ( state )
  {
  case OUTSIDE_ARRAY:

    if ( xmlStrcasecmp(name, (xmlChar *) "array") == 0 )
      read_array_tag(name, attrs);
    break;

  case INSIDE_ARRAY:

    if ( xmlStrcasecmp(name, (xmlChar *) "element") == 0 )
      read_entry(name, attrs);
    break;

  default:
    break;
  }
}
//-----------------------------------------------------------------------------
void XMLArray::end_element(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_ARRAY:

    if ( xmlStrcasecmp(name, (xmlChar *) "array") == 0 )
    {
      state = ARRAY_DONE;
      release();
    }
    break;

  default:
    break;
  }
}
//-----------------------------------------------------------------------------
void XMLArray::write(const std::vector<int>& x, uint offset,
                     std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<array type=\"int\" size=\"" << x.size() << "\">" << std::endl;
  ++indent;
  for (uint i = 0; i < x.size(); ++i)
    outfile << indent() << "<element index=\"" << i + offset << "\" value=\"" << x[i] << "\"/>" << std::endl;
  --indent;
  outfile << indent() << "</array>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLArray::write(const std::vector<uint>& x, uint offset,
                     std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<array type=\"uint\" size=\"" << x.size() << "\">" << std::endl;
  ++indent;
  for (uint i = 0; i < x.size(); ++i)
    outfile << indent() << "<element index=\"" << i + offset << "\" value=\"" << x[i] << "\"/>" << std::endl;
  --indent;
  outfile << indent() << "</array>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLArray::write(const std::vector<double>& x, uint offset,
                     std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<array type=\"double\" size=\"" << x.size() << "\">" << std::endl;
  ++indent;
  for (uint i = 0; i < x.size(); ++i)
    outfile << indent() << "<element index=\"" << i + offset << "\" value=\"" << std::setprecision(16) << x[i] << "\"/>" << std::endl;
  --indent;
  outfile << indent() << "</array>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLArray::write(const Array<double>& x, uint offset,
                     std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<array type=\"double\" size=\"" << x.size() << "\">" << std::endl;
  ++indent;
  for (uint i = 0; i < x.size(); ++i)
    outfile << indent() << "<element index=\"" << i + offset << "\" value=\"" << std::setprecision(16) << x[i] << "\"/>" << std::endl;
  --indent;
  outfile << indent() << "</array>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLArray::read_array_tag(const xmlChar* name, const xmlChar** attrs)
{
  state = INSIDE_ARRAY;

  // Parse size of array
  const uint size = parse_uint(name, attrs, "size");
  indices.clear();
  indices.reserve(size);

  std::string array_type = parse_string(name, attrs, "type");

  // Initialize vector
  switch ( atype )
  {
    case INT:
      assert(ix);
      if (!array_type.compare("int") == 0)
        error("Array file of type '%s', expected 'int'.", array_type.c_str());
      ix->clear();
      ix->reserve(size);
      break;

    case UINT:
      assert(ux);
      if (! array_type.compare("uint") == 0 )
        error("Array file of type '%s', expected 'uint'.", array_type.c_str());
      ux->clear();
      ux->reserve(size);
      break;

    case DOUBLE:
      assert(dx);
      if (! array_type.compare("double") == 0 )
        error("Array file of type '%s', expected 'double'.", array_type.c_str());
      dx->clear();
      dx->reserve(size);
      break;

    default:
      break;
  }
}
//-----------------------------------------------------------------------------
void XMLArray::read_entry(const xmlChar* name, const xmlChar** attrs)
{
  // Parse index
  const uint index = parse_uint(name, attrs, "index");
  indices.push_back(index);

  /*
  // Check values
  if (index >= global_size)
  {
    error("Illegal XML data for Array: row index %d out of range (0 - %d)",
          index, global_size - 1);
  }
  */

  int ivalue = 0;
  uint uvalue = 0;
  double dvalue = 0.0;

  // Parse value and insert in array
  switch ( atype )
  {
    case INT:
      assert(ix);
      ivalue = parse_int(name, attrs, "value");
      ix->push_back(ivalue);
      break;

     case UINT:
      assert(ux);
      uvalue = parse_uint(name, attrs, "value");
      ux->push_back(uvalue);
      break;

     case DOUBLE:
      assert(dx);
      dvalue = parse_float(name, attrs, "value");
      dx->push_back(dvalue);
      break;

     default:
      break;
  }
}
//-----------------------------------------------------------------------------
