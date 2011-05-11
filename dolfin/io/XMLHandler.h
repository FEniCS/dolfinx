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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-03-03
// Last changed: 2009-08-12

#ifndef __XMLHANDLER_H
#define __XMLHANDLER_H

#include <iomanip>
#include <fstream>
#include <libxml/parser.h>
#include <libxml/xmlreader.h>
#include <libxml/relaxng.h>
#include <dolfin/common/types.h>

namespace dolfin
{

  class XMLFile;

  class XMLHandler
  {
  public:

    /// Constructor
    XMLHandler(XMLFile& parser);

    /// Destructor
    virtual ~XMLHandler();

    void handle();

    void release();

    /// Callback for start of XML element
    virtual void start_element(const xmlChar* name, const xmlChar** attrs) = 0;

    /// Callback for end of XML element
    virtual void end_element(const xmlChar* name) = 0;

    void open_file(std::string filename);

    void open_close();

  protected:

    // Parse an integer value
    int parse_int(const xmlChar* name, const xmlChar** attrs,
                  const char *attribute);

    // Parse an unsigned integer value
    uint parse_uint(const xmlChar* name, const xmlChar** attrs,
                    const char *attribute);

    // Parse a double value
    double parse_float(const xmlChar* name, const xmlChar** attrs,
                       const char* attribute);

    // Parse a string
    std::string parse_string(const xmlChar* name, const xmlChar** attrs,
                             const char* attribute);

    // Parse a string with some forgiveness!
    std::string parse_string_optional(const xmlChar* name,
                                      const xmlChar** attrs,
                                      const char* attribute);

    // Parse a bool
    bool parse_bool(const xmlChar* name, const xmlChar** attrs,
                    const char* attribute);

    XMLFile& parser;
    std::ofstream outfile;

  };

}
#endif
