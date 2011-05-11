// Copyright (C) 2002-2006 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2002-12-06
// Last changed: 2006-10-16

#include <cstring>
#include <dolfin/log/dolfin_log.h>
#include "XMLObject.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLObject::XMLObject()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLObject::~XMLObject()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int XMLObject::parse_int(const xmlChar* name, const xmlChar** attrs,
			const char* attribute)
{
  // Check that we got the data
  if ( !attrs )
    error("Missing attribute \"%s\" for <%s> in XML file.",
                  attribute, name);

  // Parse data
  for (uint i = 0; attrs[i]; i++)
  {
    // Check for attribute
    if ( xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0 )
    {
      if ( !attrs[i+1] )
        error("Value for attribute \"%s\" of <%s> missing in XML file.",
		      attribute, name);

      int value = atoi((const char *) (attrs[i+1]));
      return value;
    }
  }

  // Didn't get the value
  error("Missing attribute \"%s\" for <%s> in XML file.",
		attribute, name);

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint XMLObject::parseUnsignedInt(const xmlChar* name,
					 const xmlChar** attrs,
					 const char* attribute)
{
  // Check that we got the data
  if ( !attrs )
    error("Missing attribute \"%s\" for <%s> in XML file.",
                  attribute, name);

  // Parse data
  for (uint i = 0; attrs[i]; i++)
  {
    // Check for attribute
    if ( xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0 )
    {
      if ( !attrs[i+1] )
        error("Value for attribute \"%s\" of <%s> missing in XML file.",
		      attribute, name);

      int value = atoi((const char *) (attrs[i+1]));
      if ( value < 0 )
      {
	error("Value for attribute \"%s\" of <%s> is negative.",
		      attribute, name);
      }
      return static_cast<uint>(value);
    }
  }

  // Didn't get the value
  error("Missing attribute \"%s\" for <%s> in XML file.",
		attribute, name);

  return 0;
}
//-----------------------------------------------------------------------------
double XMLObject::parse_real(const xmlChar* name, const xmlChar** attrs,
			  const char* attribute)
{
  // Check that we got the data
  if ( !attrs )
    error("Missing attribute \"%s\" for <%s> in XML file.",
                  attribute, name);

  // Parse data
  for (uint i = 0; attrs[i]; i++)
  {
    // Check for attribute
    if ( xmlStrcasecmp(attrs[i],(xmlChar *) attribute) == 0 )
    {
      if ( !attrs[i+1] )
        error("Value for attribute \"%s\" of <%s>  missing in XML file.",
		      attribute, name);

      double value = static_cast<double>(atof((const char *) (attrs[i+1])));
      return value;
    }
  }

  // Didn't get the value
  error("Missing attribute \"%s\" for <%s> in XML file.",
		attribute, name);

  return 0.0;
}
//-----------------------------------------------------------------------------
std::string XMLObject::parse_string(const xmlChar* name, const xmlChar** attrs,
				   const char* attribute)
{
  // Check that we got the data
  if ( !attrs )
    error("Missing attribute \"%s\" for <%s> in XML file.  No attribute list given.",
                  attribute, name);

  // Parse data
  for (uint i = 0; attrs[i]; i++)
  {
    // Check for attribute
    if ( xmlStrcasecmp(attrs[i],(xmlChar *) attribute) == 0 )
    {
      if ( !attrs[i+1] )
        error("Value for attribute \"%s\" of <%s> missing in XML file.",
		      attribute, name);

      std::string value = (const char *) (attrs[i+1]);
      return value;
    }
  }

  // Didn't get the value
  error("Missing attribute value for \"%s\" for <%s> in XML file.",
		attribute, name);

  return "";
}
//-----------------------------------------------------------------------------
std::string XMLObject::parse_stringOptional(const xmlChar* name, const xmlChar** attrs,
				   const char* attribute)
{
  // Check that we got the data
  if ( !attrs )
    error("Missing attribute \"%s\" for <%s> in XML file.  No attribute list given.",
                  attribute, name);

  // Parse data
  for (uint i = 0; attrs[i]; i++)
  {
    // Check for attribute
    if ( xmlStrcasecmp(attrs[i],(xmlChar *) attribute) == 0 )
    {
      if ( !attrs[i+1] )
        error("Value for attribute \"%s\" of <%s> missing in XML file.",
		      attribute, name);

      std::string value = (const char *) (attrs[i+1]);
      return value;
    }
  }

  // Didn't get the value, then return an empty string
  // a default will be set in the calling function
  return "";
}
//-----------------------------------------------------------------------------
bool XMLObject::parse_bool(const xmlChar* name, const xmlChar** attrs,
			const char* attribute)
{
  // Check that we got the data
  if ( !attrs )
    error("Missing attribute \"%s\" for <%s> in XML file.",
                  attribute, name);

  // Parse data
  for (uint i = 0; attrs[i]; i++)
  {
    // Check for attribute
    if ( xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0 )
    {
      if ( !attrs[i+1] )
        error("Value for attribute \"%s\" of <%s> missing in XML file.",
		      attribute, name);

      std::string value = (const char *) (attrs[i+1]);
      if ( strcmp(value.c_str(), "true") == 0 or strcmp(value.c_str(), "1") == 0 )
        return true;
      if ( strcmp(value.c_str(), "false") == 0 or strcmp(value.c_str(), "0") == 0 )
        return false;

      error("Cannot convert \"%s\" for attribute \"%s\" in <%s> to bool.",
		    value.c_str(), attribute, name);
      return false;

    }
  }

  // Didn't get the value
  error("Missing attribute \"%s\" for <%s> in XML file.",
		attribute, name);

  return 0;
}

//-----------------------------------------------------------------------------
void XMLObject::open(std::string filename)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool XMLObject::close()
{
  return true;
}
//-----------------------------------------------------------------------------
