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
// Modified by Anders Logg, 2011.
//
// First added:  2009-03-03
// Last changed: 2011-03-31

#ifndef __OLDXMLFILE_H
#define __OLDXMLFILE_H

#include <fstream>
#include <map>
#include <string>
#include <stack>
#include <vector>

#include <libxml/parser.h>

#include <dolfin/common/MPI.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/plot/FunctionPlotData.h>
#include "GenericFile.h"
#include "XMLLocalMeshDataDistributed.h"
#include "XMLFunctionPlotData.h"
#include "XMLDolfin.h"
#include "XMLHandler.h"

namespace dolfin
{

  class GenericVector;
  class Mesh;
  class Parameters;

  class OldXMLFile: public GenericFile
  {
  public:

    /// Constructor
    OldXMLFile(const std::string filename);

    /// Constructor from a stream
    OldXMLFile(std::ostream& s);

    /// Destructor
    ~OldXMLFile();

    void operator>> (LocalMeshData& input)
    { read_xml(input); }

    /// Parse file
    void parse();

    /// Push handler onto stack
    void push(XMLHandler* handler);

    /// Pop handler from stack
    void pop();

    /// Return handler from top of stack
    XMLHandler* top();

  private:

    // Friends
    friend void sax_start_element(void *ctx, const xmlChar *name, const xmlChar **attrs);
    friend void sax_end_element(void *ctx, const xmlChar *name);

    // Read XML data
    template<class T> void read_xml(T& t)
    {
      typedef typename T::XMLHandler Handler;
      Handler xml_handler(t, *this);
      XMLDolfin xml_dolfin(xml_handler, *this);
      xml_dolfin.handle();

      parse();

      if (!handlers.empty())
        error("Handler stack not empty. Something is wrong!");
    }

    std::stack<XMLHandler*> handlers;
    xmlSAXHandler* sax;
    std::ostream* outstream;

    void start_element(const xmlChar *name, const xmlChar **attrs);
    void end_element  (const xmlChar *name);

    void open_file();
    void close_file();

  };

  // Callback functions for the SAX interface

  void sax_start_document (void *ctx);
  void sax_end_document   (void *ctx);
  void sax_start_element  (void *ctx, const xmlChar *name, const xmlChar **attrs);
  void sax_end_element    (void *ctx, const xmlChar *name);

  void sax_warning     (void *ctx, const char *msg, ...);
  void sax_error       (void *ctx, const char *msg, ...);
  void sax_fatal_error (void *ctx, const char *msg, ...);

  // Callback functions for Relax-NG Schema

  void rng_parser_error(void *user_data, xmlErrorPtr error);
  void rng_valid_error (void *user_data, xmlErrorPtr error);

}
#endif
