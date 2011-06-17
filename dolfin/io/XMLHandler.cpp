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
// Last changed: 2009-03-04

#include <cstring>
#include <sstream>
#include <fstream>
#include <dolfin/log/log.h>
#include "XMLFile.h"
#include "XMLHandler.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLHandler::XMLHandler(XMLFile& parser) : parser(parser)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLHandler::~XMLHandler()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLHandler::handle()
{
  parser.push(this);
}
//-----------------------------------------------------------------------------
void XMLHandler::release()
{
  parser.pop();
}
//-----------------------------------------------------------------------------
/*
void XMLHandler::validate(std::string filename)
{
  xmlTextReaderPtr reader;
  xmlRelaxNGParserCtxtPtr rngp;
  xmlRelaxNGPtr rngs;
  int ret;
  reader = xmlNewTextReaderFilename(filename.c_str());
  rngp = xmlRelaxNGNewParserCtxt("test.rng");
  rngs = xmlRelaxNGParse(rngp);
  xmlTextReaderRelaxNGSetSchema(reader, rngs);
  //char* schema;
  //std::ifstream schema_file;
  //schema_file.open("test.rng");
  //length = schema_file.tellg();
  //schema = new char[length];
  //schema_file.read(schema, length);
  //schema_file.close();
  if ( reader != NULL ) {
    ret = xmlTextReaderRead(reader);
    while ( ret == 1 ) {
      ret = xmlTextReaderRead(reader);
      if ( ret != 0 ) {
        error("%s : failed to parse\n", filename.c_str());
      }
    }
  }
  else {
    error("Unable to open %s\n", filename.c_str());
  }
  if ( xmlTextReaderIsValid(reader) == true ) {
    error("%s validates", filename.c_str());
  }
  else {
    error("%s fails to validate", filename.c_str());
  }
}
*/
//-----------------------------------------------------------------------------
void XMLHandler::open_file(std::string filename)
{
  // Open file
  outfile.open(filename.c_str());

  // Go to end of file
  outfile.seekp(0, std::ios::end);
}
//-----------------------------------------------------------------------------
int XMLHandler::parse_int(const xmlChar* name, const xmlChar** attrs,
			const char* attribute)
{
  // Check that we got the data
  if ( !attrs )
    error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);

  // Parse data
  for (uint i = 0; attrs[i]; i++)
  {
    // Check for attribute
    if ( xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0 )
    {
      if ( !attrs[i+1] )
      {
        error("Value for attribute \"%s\" of <%s> missing in XML file.",
     		      attribute, name);
      }

      std::istringstream ss((const char *)attrs[i+1]);
      int value;
      ss >> value;
      return value;
    }
  }

  // Didn't get the value
  error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint XMLHandler::parse_uint(const xmlChar* name,
					 const xmlChar** attrs,
					 const char* attribute)
{
  // Check that we got the data
  if ( !attrs )
    error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);

  // Parse data
  for (uint i = 0; attrs[i]; i++)
  {
    // Check for attribute
    if ( xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0 )
    {
      if ( !attrs[i+1] )
      {
        error("Value for attribute \"%s\" of <%s> missing in XML file.",
     		      attribute, name);
      }

      std::istringstream ss((const char *)attrs[i+1]);
      int value;
      ss >> value;
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
double XMLHandler::parse_float(const xmlChar* name, const xmlChar** attrs,
			  const char* attribute)
{
  // Check that we got the data
  if ( !attrs )
    error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);

  // Parse data
  for (uint i = 0; attrs[i]; i++)
  {
    // Check for attribute
    if ( xmlStrcasecmp(attrs[i],(xmlChar *) attribute) == 0 )
    {
      if ( !attrs[i+1] )
      {
        error("Value for attribute \"%s\" of <%s>  missing in XML file.",
    		      attribute, name);
      }

      std::istringstream ss((const char *)attrs[i+1]);
      double value;
      ss >> value;
      return value;
    }
  }

  // Didn't get the value
  error("Missing attribute \"%s\" for <%s> in XML file.",
		attribute, name);

  return 0.0;
}
//-----------------------------------------------------------------------------
std::string XMLHandler::parse_string(const xmlChar* name, const xmlChar** attrs,
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
std::string XMLHandler::parse_string_optional(const xmlChar* name,
                                              const xmlChar** attrs,
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
bool XMLHandler::parse_bool(const xmlChar* name, const xmlChar** attrs,
			                      const char* attribute)
{
  // Check that we got the data
  if ( !attrs )
    error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);

  // Parse data
  for (uint i = 0; attrs[i]; i++)
  {
    // Check for attribute
    if ( xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0 )
    {
      if ( !attrs[i+1] )
      {
        error("Value for attribute \"%s\" of <%s> missing in XML file.",
		      attribute, name);
      }

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
  error("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);

  return 0;
}
//-----------------------------------------------------------------------------
