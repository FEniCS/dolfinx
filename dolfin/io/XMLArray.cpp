// Copyright (C) 2009 Ola Skavhaug
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-03-02
// Last changed: 2009-03-17

#include <dolfin/common/Array.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include "XMLFile.h"
#include "XMLIndent.h"
#include "XMLArray.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLArray::XMLArray(std::vector<int>& ix, XMLFile& parser, bool distributed)
  : XMLHandler(parser), ix(&ix), ux(0), dx(0), state(OUTSIDE_ARRAY),
    atype(INT), size(0), distributed(distributed)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLArray::XMLArray(std::vector<uint>& ux, XMLFile& parser, bool distributed)
  : XMLHandler(parser), ix(0), ux(&ux), dx(0), state(OUTSIDE_ARRAY),
    atype(UINT), size(0), distributed(distributed)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLArray::XMLArray(std::vector<double>& dx, XMLFile& parser, bool distributed)
  : XMLHandler(parser), ix(0), ux(0), dx(&dx), state(OUTSIDE_ARRAY),
    atype(DOUBLE), size(0), distributed(distributed)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLArray::XMLArray(std::vector<int>& ix, XMLFile& parser, uint size, bool distributed)
  : XMLHandler(parser), ix(&ix), ux(0), dx(0), state(INSIDE_ARRAY), atype(INT),
    size(size), distributed(distributed)
{
  if (distributed)
    range = MPI::local_range(size);
  else
    range = std::make_pair(0, size);
  const uint local_size = range.second - range.first;

  element_index.reserve(local_size);
  this->ix->clear();
  this->ix->reserve(local_size);
}
//-----------------------------------------------------------------------------
XMLArray::XMLArray(std::vector<uint>& ux, XMLFile& parser, uint size, bool distributed)
  : XMLHandler(parser), ix(0), ux(&ux), dx(0), state(INSIDE_ARRAY), atype(UINT),
    size(size), distributed(distributed)
{
  if (distributed)
    range = MPI::local_range(size);
  else
    range = std::make_pair(0, size);
  const uint local_size = range.second - range.first;

  element_index.reserve(local_size);
  this->ux->clear();
  this->ux->reserve(local_size);
}
//-----------------------------------------------------------------------------
XMLArray::XMLArray(std::vector<double>& dx, XMLFile& parser, uint size, bool distributed)
  : XMLHandler(parser), ix(0), ux(0), dx(&dx), state(INSIDE_ARRAY),
    atype(DOUBLE), size(size), distributed(distributed)
{
  range = MPI::local_range(size);
  const uint local_size = range.second - range.first;

  element_index.reserve(local_size);
  this->dx->clear();
  this->dx->reserve(local_size);
}
//-----------------------------------------------------------------------------
void XMLArray::start_element(const xmlChar *name, const xmlChar **attrs)
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
void XMLArray::read_array_tag(const xmlChar *name, const xmlChar **attrs)
{
  state = INSIDE_ARRAY;

  // Parse size of array
  size = parse_uint(name, attrs, "size");

  if (distributed)
    range = MPI::local_range(size);
  else
    range = std::make_pair(0, size);
  const uint local_size = range.second - range.first;

  const std::string array_type = parse_string(name, attrs, "type");

  // Initialize index array
  element_index.clear();
  element_index.reserve(local_size);

  // Initialize data array
  switch ( atype )
  {
    case INT:
      assert(ix);
      if (!array_type.compare("int") == 0)
        error("Array file of type '%s', expected 'int'.", array_type.c_str());
      ix->clear();
      ix->reserve(local_size);
      break;

    case UINT:
      assert(ux);
      if (! array_type.compare("uint") == 0 )
        error("Array file of type '%s', expected 'uint'.", array_type.c_str());
      ux->clear();
      ux->reserve(local_size);
      break;

    case DOUBLE:
      assert(dx);
      if (! array_type.compare("double") == 0 )
        error("Array file of type '%s', expected 'double'.", array_type.c_str());
      dx->clear();
      dx->reserve(local_size);
      break;

    default:
      break;
  }
}
//-----------------------------------------------------------------------------
void XMLArray::read_entry(const xmlChar *name, const xmlChar **attrs)
{
  // Parse index
  const uint index = parse_uint(name, attrs, "index");

  // Check values
  if (index >= size)
  {
    error("Illegal XML data for Array: row index %d out of range (0 - %d)",
          index, size - 1);
  }

  if (index >= range.first && index < range.second)
  {
    element_index.push_back(index);

    // Parse value and insert in data array
    switch ( atype )
    {
      case INT:
        assert(ix);
        ix->push_back(parse_int(name, attrs, "value"));
        break;

      case UINT:
        assert(ux);
        ux->push_back(parse_uint(name, attrs, "value"));
        break;

      case DOUBLE:
        assert(dx);
        dx->push_back(parse_float(name, attrs, "value"));
        break;

      default:
        break;
    }
  }
}
//-----------------------------------------------------------------------------
