// Copyright (C) 2009 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-03
// Last changed: 2009-03-04

#include <dolfin/log/dolfin_log.h>
#include "XMLArray.h"
#include "XMLMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLMap::XMLMap(std::map<uint, int>& im, NewXMLFile& parser)
  : XMLHandler(parser), im(&im), um(0), dm(0), iam(0), uam(0), dam(0), state(OUTSIDE_MAP), mtype(INT), current_key(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLMap::XMLMap(std::map<uint, uint>& um, NewXMLFile& parser)
  : XMLHandler(parser), im(0), um(&um), dm(0), iam(0), uam(0), dam(0), state(OUTSIDE_MAP), mtype(INT), current_key(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLMap::XMLMap(std::map<uint, double>& dm, NewXMLFile& parser)
  : XMLHandler(parser), im(0), um(0), dm(&dm), iam(0), uam(0), dam(0), state(OUTSIDE_MAP), mtype(INT), current_key(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLMap::XMLMap(std::map<uint, std::vector<int> >& iam, NewXMLFile& parser)
  : XMLHandler(parser), im(0), um(0), dm(0), iam(&iam), uam(0), dam(0), state(OUTSIDE_MAP), mtype(INT_ARRAY), current_key(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLMap::XMLMap(std::map<uint, std::vector<uint> >& uam, NewXMLFile& parser)
  : XMLHandler(parser), im(0), um(0), dm(0), iam(0), uam(&uam), dam(0), state(OUTSIDE_MAP), mtype(UINT_ARRAY), current_key(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLMap::XMLMap(std::map<uint, std::vector<double> >& dam, NewXMLFile& parser)
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
  dolfin_assert( key_type.compare("uint") == 0 ); 
  
  // Initialize map
  switch ( mtype )
  {
    case INT:
      dolfin_assert( value_type.compare("int") == 0 );
      dolfin_assert(im);
      im->clear();
      
      break;

    case UINT:
      dolfin_assert( value_type.compare("uint") == 0 );
      dolfin_assert(um);
      um->clear();

      break;

    case DOUBLE:
      dolfin_assert( value_type.compare("double") == 0 );
      dolfin_assert(dm);
      dm->clear();

      break;

    case INT_ARRAY:
      dolfin_assert( value_type.compare("array") == 0 );
      dolfin_assert(iam);
      iam->clear();

      break;

    case UINT_ARRAY:
      dolfin_assert( value_type.compare("array") == 0 );
      dolfin_assert(uam);
      uam->clear();

      break;

    case DOUBLE_ARRAY:
      dolfin_assert( value_type.compare("array") == 0 );
      dolfin_assert(dam);
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
  (*im)[current_key] = parse_int(name, attrs, "value");
}
//-----------------------------------------------------------------------------
void XMLMap::read_uint(const xmlChar *name, const xmlChar **attrs)
{
  (*um)[current_key] = parse_uint(name, attrs, "value");
}
//-----------------------------------------------------------------------------
void XMLMap::read_double(const xmlChar *name, const xmlChar **attrs)
{
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
