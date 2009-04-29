// Copyright (C) 2009 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-03
// Last changed: 2009-03-04

#ifndef __XMLHANDLER_H
#define __XMLHANDLER_H

#include <iomanip>
#include <fstream>
#include <libxml/parser.h>
#include <libxml/xmlreader.h>
#include <libxml/relaxng.h>

namespace dolfin
{

  class NewXMLFile;

  class XMLHandler
  {
  public:

    /// Constructor
    XMLHandler(NewXMLFile& parser);

    /// Destructor
    virtual ~XMLHandler();

    void handle();

    void release();

    //void validate(std::string filename);

    /// Callback for start of XML element
    virtual void start_element(const xmlChar* name, const xmlChar** attrs) = 0;

    /// Callback for end of XML element
    virtual void end_element(const xmlChar* name) = 0;

    void open_file(std::string filename);

    void open_close();

  protected:

    // Parse an integer value
    int parse_int(const xmlChar* name, const xmlChar** attrs, const char *attribute);

    // Parse an unsigned integer value
    uint parse_uint(const xmlChar* name, const xmlChar** attrs, const char *attribute);

    // Parse a double value
    double parse_float(const xmlChar* name, const xmlChar** attrs, const char* attribute);

    // Parse a string
    std::string parse_string(const xmlChar* name, const xmlChar** attrs, const char* attribute);

    // Parse a string with some forgiveness!
    std::string parse_string_optional(const xmlChar* name, const xmlChar** attrs, const char* attribute);

    // Parse a bool
    bool parse_bool(const xmlChar* name, const xmlChar** attrs, const char* attribute);

    NewXMLFile& parser;
    std::ofstream outfile;

  };

}
#endif
