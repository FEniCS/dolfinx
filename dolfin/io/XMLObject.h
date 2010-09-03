// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-07-15
// Last changed: 2006-05-23

#ifndef __XML_OBJECT_H
#define __XML_OBJECT_H

#include <string>
#include <libxml/parser.h>
#include <dolfin/common/types.h>

namespace dolfin
{

  class XMLObject
  {
  public:

    /// Constructor
    XMLObject();

    /// Destructor
    virtual ~XMLObject();

    /// Callback for start of XML element
    virtual void start_element(const xmlChar* name, const xmlChar** attrs) = 0;

    /// Callback for end of XML element
    virtual void end_element(const xmlChar* name) = 0;

    /// Callback for start of XML file (optional)
    virtual void open(std::string filename);

    /// Callback for end of XML file, should return true iff data is ok (optional)
    virtual bool close();

  protected:

    // Parse an integer value
    int parse_int(const xmlChar* name, const xmlChar** attrs, const char *attribute);

    // Parse an unsigned integer value
    uint parseUnsignedInt(const xmlChar* name, const xmlChar** attrs, const char *attribute);

    // Parse a double value
    double parse_real(const xmlChar* name, const xmlChar** attrs, const char* attribute);

    // Parse a string
    std::string parse_string(const xmlChar* name, const xmlChar** attrs, const char* attribute);

    // Parse a string with some forgiveness!
    std::string parse_stringOptional(const xmlChar* name, const xmlChar** attrs, const char* attribute);

    // Parse a bool
    bool parse_bool(const xmlChar* name, const xmlChar** attrs, const char* attribute);

  };

}

#endif
