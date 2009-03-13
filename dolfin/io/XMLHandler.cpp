// Copyright (C) 2009 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added: 2009-03-03
// Last changed: 2009-03-04

#include <cstring>
#include <sstream>
#include <dolfin/log/log.h>
#include "NewXMLFile.h"
#include "XMLHandler.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLHandler::XMLHandler(NewXMLFile& parser) : parser(parser)
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
        error("Value for attribute \"%s\" of <%s> missing in XML file.",
		      attribute, name);
      
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
std::string XMLHandler::parse_string_optional(const xmlChar* name, const xmlChar** attrs,
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

