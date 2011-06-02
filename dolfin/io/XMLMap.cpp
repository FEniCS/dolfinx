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
// First added:  2009-03-03
// Last changed: 2009-03-17

#include <dolfin/log/dolfin_log.h>
#include "XMLIndent.h"
#include "XMLArray.h"
#include "XMLMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLMap::XMLMap(std::map<uint, int>& im, XMLFile& parser)
  : XMLHandler(parser), im(&im), um(0), dm(0), iam(0), uam(0), dam(0), state(OUTSIDE_MAP), mtype(INT), current_key(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLMap::XMLMap(std::map<uint, uint>& um, XMLFile& parser)
  : XMLHandler(parser), im(0), um(&um), dm(0), iam(0), uam(0), dam(0), state(OUTSIDE_MAP), mtype(UINT), current_key(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLMap::XMLMap(std::map<uint, double>& dm, XMLFile& parser)
  : XMLHandler(parser), im(0), um(0), dm(&dm), iam(0), uam(0), dam(0), state(OUTSIDE_MAP), mtype(DOUBLE), current_key(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLMap::XMLMap(std::map<uint, std::vector<int> >& iam, XMLFile& parser)
  : XMLHandler(parser), im(0), um(0), dm(0), iam(&iam), uam(0), dam(0), state(OUTSIDE_MAP), mtype(INT_ARRAY), current_key(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLMap::XMLMap(std::map<uint, std::vector<uint> >& uam, XMLFile& parser)
  : XMLHandler(parser), im(0), um(0), dm(0), iam(0), uam(&uam), dam(0), state(OUTSIDE_MAP), mtype(UINT_ARRAY), current_key(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLMap::XMLMap(std::map<uint, std::vector<double> >& dam, XMLFile& parser)
  : XMLHandler(parser), im(0), um(0), dm(0), iam(0), uam(0), dam(&dam), state(OUTSIDE_MAP), mtype(DOUBLE_ARRAY), current_key(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLMap::start_element(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE_MAP:

    if ( xmlStrcasecmp(name, (xmlChar *) "map") == 0 )
    {
      start_map(name, attrs);
      state = INSIDE_MAP;
    }

    break;

  case INSIDE_MAP:

    if ( xmlStrcasecmp(name, (xmlChar *) "map_entry") == 0 )
    {
      read_map_entry(name, attrs);
      state = INSIDE_MAP_ENTRY;
    }

    break;

  case INSIDE_MAP_ENTRY:
    if ( xmlStrcasecmp(name, (xmlChar *) "int") == 0 )
      read_int(name, attrs);
    if ( xmlStrcasecmp(name, (xmlChar *) "uint") == 0 )
      read_uint(name, attrs);
    if ( xmlStrcasecmp(name, (xmlChar *) "double") == 0 )
      read_double(name, attrs);
    if ( xmlStrcasecmp(name, (xmlChar *) "array") == 0 )
      read_array(name, attrs);

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMap::end_element(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_MAP:

    if ( xmlStrcasecmp(name, (xmlChar *) "map") == 0 )
    {
      state = MAP_DONE;
      release();
    }

    break;

  case INSIDE_MAP_ENTRY:
    if ( xmlStrcasecmp(name, (xmlChar *) "map_entry") == 0 )
    {
      finalize_map_entry();
      state = INSIDE_MAP;
    }

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMap::write(const std::map<uint, int>& map, std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<map key_type=\"uint\" value_type=\"int\">" << std::endl;

  ++indent;
  std::map<uint, int>::const_iterator pos;
  for (pos = map.begin(); pos != map.end(); ++pos)
  {
    outfile << indent() << "<map_entry key=\"" << (*pos).first << "\">" << std::endl;
    ++indent;
    outfile << indent() << "<int value=\"" << (*pos).second << "\"/>" << std::endl;
    --indent;
    outfile << indent() << "</map_entry>" << std::endl;
  }
  --indent;
  outfile << indent() << "</map>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLMap::write(const std::map<uint, uint>& map, std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<map key_type=\"uint\" value_type=\"uint\">" << std::endl;

  ++indent;
  std::map<uint, uint>::const_iterator pos;
  for (pos = map.begin(); pos != map.end(); ++pos)
  {
    outfile << indent() << "<map_entry key=\"" << (*pos).first << "\">" << std::endl;
    ++indent;
    outfile << indent() << "<uint value=\"" << (*pos).second << "\"/>" << std::endl;
    --indent;
    outfile << indent() << "</map_entry>" << std::endl;
  }
  --indent;
  outfile << indent() << "</map>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLMap::write(const std::map<uint, double>& map, std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<map key_type=\"uint\" value_type=\"double\">" << std::endl;

  ++indent;
  std::map<uint, double>::const_iterator pos;
  for (pos = map.begin(); pos != map.end(); ++pos)
  {
    outfile << indent() << "<map_entry key=\"" << (*pos).first << "\">" << std::endl;
    ++indent;
    outfile << indent() << "<double value=\"" << std::setprecision(16) << (*pos).second << "\"/>" << std::endl;
    --indent;
    outfile << indent() << "</map_entry>" << std::endl;
  }
  --indent;
  outfile << indent() << "</map>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLMap::write(const std::map<uint, std::vector<int> >& map, std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<map key_type=\"uint\" value_type=\"array\">" << std::endl;

  ++indent;
  std::map<uint, std::vector<int> >::const_iterator pos;
  for (pos = map.begin(); pos != map.end(); ++pos)
  {
    outfile << indent() << "<map_entry key=\"" << (*pos).first << "\">" << std::endl;
    ++indent;
    XMLArray::write((*pos).second, 0, outfile, indent.level());
    --indent;
    outfile << indent() << "</map_entry>" << std::endl;
  }
  --indent;
  outfile << indent() << "</map>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLMap::write(const std::map<uint, std::vector<uint> >& map, std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<map key_type=\"uint\" value_type=\"array\">" << std::endl;

  ++indent;
  std::map<uint, std::vector<uint> >::const_iterator pos;
  for (pos = map.begin(); pos != map.end(); ++pos)
  {
    outfile << indent() << "<map_entry key=\"" << (*pos).first << "\">" << std::endl;
    ++indent;
    XMLArray::write((*pos).second, 0, outfile, indent.level());
    --indent;
    outfile << indent() << "</map_entry>" << std::endl;
  }
  --indent;
  outfile << indent() << "</map>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLMap::write(const std::map<uint, std::vector<double> >& map, std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<map key_type=\"uint\" value_type=\"array\">" << std::endl;

  ++indent;
  std::map<uint, std::vector<double> >::const_iterator pos;
  for (pos = map.begin(); pos != map.end(); ++pos)
  {
    outfile << indent() << "<map_entry key=\"" << (*pos).first << "\">" << std::endl;
    ++indent;
    XMLArray::write((*pos).second, 0, outfile, indent.level());
    --indent;
    outfile << indent() << "</map_entry>" << std::endl;
  }
  --indent;
  outfile << indent() << "</map>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLMap::finalize_map_entry()
{
  switch ( mtype )
  {
    case INT_ARRAY:
      delete xml_array;
      (*iam)[current_key] = (*ix); // Copy array
      delete ix;
      ix = 0;

      break;

    case UINT_ARRAY:
      delete xml_array;
      (*uam)[current_key] = (*ux); // Copy array
      delete ux;
      ux = 0;

      break;

    case DOUBLE_ARRAY:
      delete xml_array;
      (*dam)[current_key] = (*dx); // Copy array
      delete dx;
      dx = 0;

      break;

    default:
      ; // Do nothing
  }
}
//-----------------------------------------------------------------------------
void XMLMap::start_map(const xmlChar *name, const xmlChar **attrs)
{
  // Parse key type
  std::string key_type = parse_string(name, attrs, "key_type");

  // Parse valuetype
  std::string value_type = parse_string(name, attrs, "value_type");

  // Make sure key is an unsigned integer
  assert( key_type.compare("uint") == 0 );

  // Initialize map
  switch ( mtype )
  {
    case INT:
      assert( value_type.compare("int") == 0 );
      assert(im);
      im->clear();

      break;

    case UINT:
      assert( value_type.compare("uint") == 0 );
      assert(um);
      um->clear();

      break;

    case DOUBLE:
      assert( value_type.compare("double") == 0 );
      assert(dm);
      dm->clear();

      break;

    case INT_ARRAY:
      assert( value_type.compare("array") == 0 );
      assert(iam);
      iam->clear();

      break;

    case UINT_ARRAY:
      assert( value_type.compare("array") == 0 );
      assert(uam);
      uam->clear();

      break;

    case DOUBLE_ARRAY:
      assert( value_type.compare("array") == 0 );
      assert(dam);
      dam->clear();

      break;

    default:
      ;
  }
}
//-----------------------------------------------------------------------------
void XMLMap::read_map_entry(const xmlChar *name, const xmlChar **attrs)
{
  // Parse index
  current_key = parse_uint(name, attrs, "key");
}
//-----------------------------------------------------------------------------
void XMLMap::read_int(const xmlChar *name, const xmlChar **attrs)
{
  assert(im);
  if (! mtype == INT)
    error("Map of value type 'double' initialized, but value type in file is not.");
  (*im)[current_key] = parse_int(name, attrs, "value");
}
//-----------------------------------------------------------------------------
void XMLMap::read_uint(const xmlChar *name, const xmlChar **attrs)
{
  assert(um);
  if (! mtype == UINT )
    error("Map of value type 'uint' initialized, but value type in file is not.");
  (*um)[current_key] = parse_uint(name, attrs, "value");
}
//-----------------------------------------------------------------------------
void XMLMap::read_double(const xmlChar *name, const xmlChar **attrs)
{
  assert(dm);
  if (! mtype == DOUBLE)
    error("Map of value type 'double' initialized, but value type in file is not.");
  (*dm)[current_key] = parse_float(name, attrs, "value");
}
//-----------------------------------------------------------------------------
void XMLMap::read_array(const xmlChar *name, const xmlChar **attrs)
{
  uint size = parse_uint(name, attrs, "size");


  switch ( mtype )
  {
    case INT_ARRAY:
      read_int_array(name, attrs, size);

      break;

    case UINT_ARRAY:
      read_uint_array(name, attrs, size);

      break;

    case DOUBLE_ARRAY:
      read_double_array(name, attrs, size);

      break;

    default:
      ;

  }

}
//-----------------------------------------------------------------------------
void XMLMap::read_int_array(const xmlChar *name, const xmlChar **attrs, uint size)
{
  std::string array_type = parse_string(name, attrs, "type");
  if ( !array_type.compare("int") == 0 )
    error("Map with arrays of type '%s', expected 'int'.", array_type.c_str());
  ix = new std::vector<int>();
  xml_array = new XMLArray(*ix, parser, size);
  xml_array->handle();
}
//-----------------------------------------------------------------------------
void XMLMap::read_uint_array(const xmlChar *name, const xmlChar **attrs, uint size)
{
  std::string array_type = parse_string(name, attrs, "type");
  if ( !array_type.compare("uint") == 0 )
    error("Map with arrays of type '%s', expected 'uint'.", array_type.c_str());
  ux = new std::vector<uint>();
  xml_array = new XMLArray(*ux, parser, size);
  xml_array->handle();
}
//-----------------------------------------------------------------------------
void XMLMap::read_double_array(const xmlChar *name, const xmlChar **attrs, uint size)
{
  std::string array_type = parse_string(name, attrs, "type");
  if ( !array_type.compare("double") == 0 )
    error("Map with arrays of type '%s', expected 'double'.", array_type.c_str());
  dx = new std::vector<double>();
  xml_array = new XMLArray(*dx, parser, size);
  xml_array->handle();
}
//-----------------------------------------------------------------------------
