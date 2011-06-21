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

#ifndef __XMLFILE_H
#define __XMLFILE_H

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
#include "XMLArray.h"
#include "XMLMap.h"
#include "XMLLocalMeshData.h"
#include "XMLFunctionPlotData.h"
#include "XMLDolfin.h"
#include "XMLHandler.h"

namespace pugi
{
  class xml_document;
  class xml_node;
}


namespace dolfin
{

  class GenericVector;
  class Mesh;
  class Parameters;

  class XMLFile: public GenericFile
  {
  public:

    /// Constructor
    XMLFile(const std::string filename, bool gzip);

    /// Constructor from a stream
    XMLFile(std::ostream& s);

    /// Destructor
    ~XMLFile();

    // Input/output handling

    // Mesh
    void operator>> (Mesh& input);
    void operator<< (const Mesh& output);

    void operator>> (LocalMeshData& input)
    { read_xml(input); }

    // Vector
    void operator>> (GenericVector& input);
    void operator<< (const GenericVector& output);

    // Parameters
    void operator>> (Parameters& input);
    void operator<< (const Parameters& output);

    // FunctionPlotData
    void operator>> (FunctionPlotData& input);
    void operator<< (const FunctionPlotData& output);

    void operator>> (MeshFunction<int>& input)
    { read_mesh_function(input, "int"); }
    void operator<< (const MeshFunction<int>& output)
    { write_mesh_function(output, "int"); }

    void operator>> (MeshFunction<unsigned int>& input)
    { read_mesh_function(input, "uint"); }
    void operator<< (const MeshFunction<unsigned int>& output)
    { write_mesh_function(output, "uint"); }

    void operator>> (MeshFunction<double>& input)
    { read_mesh_function(input, "double"); }
    void operator<< (const MeshFunction<double>& output)
    { write_mesh_function(output, "double"); }

    void operator>> (MeshFunction<bool>& input)
    { read_mesh_function(input, "bool"); }
    void operator<< (const MeshFunction<bool>& input)
    { write_mesh_function(input, "bool"); }

    //void operator>> (std::vector<int>& x)                                   { read_xml_array(x); }
    //void operator>> (std::vector<uint>& x)                                  { read_xml_array(x); }
    //void operator>> (std::vector<double>& x)                                { read_xml_array(x); }
    //void operator>> (std::map<uint, int>& map)                              { read_xml_map(map); }
    //void operator>> (std::map<uint, uint>& map)                             { read_xml_map(map); }
    //void operator>> (std::map<uint, double>& map)                           { read_xml_map(map); }
    //void operator>> (std::map<uint, std::vector<int> >& array_map)          { read_xml_map(array_map); }
    //void operator>> (std::map<uint, std::vector<uint> >& array_map)         { read_xml_map(array_map); }
    //void operator>> (std::map<uint, std::vector<double> >& array_map)       { read_xml_map(array_map); }

    //--- Mappings from output to correct handler ---


    //void operator<< (const std::vector<int>& x)                             { write_xml_array(x); }
    //void operator<< (const std::vector<uint>& x)                            { write_xml_array(x); }
    //void operator<< (const std::vector<double>& x)                          { write_xml_array(x); }
    //void operator<< (const std::map<uint, int>& map)                        { write_xml_map(map); }
    //void operator<< (const std::map<uint, uint>& map)                       { write_xml_map(map); }
    //void operator<< (const std::map<uint, double>& map)                     { write_xml_map(map); }
    //void operator<< (const std::map<uint, std::vector<int> >& array_map)    { write_xml_map(array_map); }
    //void operator<< (const std::map<uint, std::vector<uint> >& array_map)   { write_xml_map(array_map); }
    //void operator<< (const std::map<uint, std::vector<double> >& array_map) { write_xml_map(array_map); }

    /// Write file
    void write();

    /// Parse file
    void parse();

    /// Push handler onto stack
    void push(XMLHandler* handler);

    /// Pop handler from stack
    void pop();

    /// Return handler from top of stack
    XMLHandler* top();

  private:

    // Write MeshFunction
    template<class T> void read_mesh_function(MeshFunction<T>& t,
                                              const std::string type) const;

    // Read MeshFunction
    template<class T> void write_mesh_function(const MeshFunction<T>& t,
                                               const std::string type);

    // Get DOLFIN XML node
    const pugi::xml_node get_dolfin_xml_node(pugi::xml_document& xml_doc,
                                             const std::string filename) const;


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

    /*
    // Read std::map from XML file (speciliased templated required
    // for STL objects)
    template<class T> void read_xml_map(T& map)
    {
      log(TRACE, "Reading map from file %s.", filename.c_str());
      XMLMap xml_map(map, *this);
      XMLDolfin xml_dolfin(xml_map, *this);
      xml_dolfin.handle();
      parse();
      if ( !handlers.empty() )
        error("Hander stack not empty. Something is wrong!");
    }
    */

    // Read std::vector from XML file (speciliased templated required
    // for STL objects)
    template<class T> void read_xml_array(T& x)
    {
      log(TRACE, "Reading array from file %s.", filename.c_str());
      XMLArray xml_array(x, *this);
      XMLDolfin xml_dolfin(xml_array, *this);
      xml_dolfin.handle();
      parse();
      if ( !handlers.empty() )
        error("Hander stack not empty. Something is wrong!");
    }

    // Template function for writing XML
    template<class T> void write_xml(const T& t, bool is_distributed=true)
    {
      // Open file on process 0 for distributed objects and on all processes
      // for local objects
      if ((is_distributed && MPI::process_number() == 0) || !is_distributed)
        open_file();

      // FIXME: 'write' is being called on all processes since collective MPI
      // FIXME: calls might be used. Should use approach to gather data on process 0.

      // Determine appropriate handler and write
      typedef typename T::XMLHandler Handler;
      Handler::write(t, *outstream, 1);

      // Close file
      if ((is_distributed && MPI::process_number() == 0) || !is_distributed)
        close_file();
    }

    /*
    template<class T> void write_xml_map(const T& map)
    {
      // FIXME: Should we support distributed std::map?
      open_file();
      XMLMap::write(map, *outstream, 1);
      close_file();
    }
    */

    template<class T> void write_xml_array(const T& x)
    {
      open_file();
      XMLArray::write(x, 0, *outstream, 1);
      close_file();
    }

    std::stack<XMLHandler*> handlers;
    xmlSAXHandler* sax;
    std::ostream* outstream;
    bool gzip;

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
