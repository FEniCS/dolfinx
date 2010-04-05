// Copyright (C) 2009 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-03
// Last changed: 2010-01-04

#ifndef __XMLFILE_H
#define __XMLFILE_H

#include <fstream>
#include <stack>
#include <string>
#include <map>
#include <vector>
#include <libxml/parser.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/plot/FunctionPlotData.h>
#include <dolfin/parameter/Parameters.h>
#include "GenericFile.h"
#include "XMLMap.h"
#include "XMLMesh.h"
#include "XMLMeshFunction.h"
#include "XMLMatrix.h"
#include "XMLLocalMeshData.h"
#include "XMLParameters.h"
#include "XMLFunctionPlotData.h"
#include "XMLDolfin.h"
#include "XMLHandler.h"

namespace dolfin
{

  class XMLFile: public GenericFile
  {
  public:

    /// Constructor
    XMLFile(const std::string filename, bool gzip);

    XMLFile(std::ostream& s);

    /// Destructor
    ~XMLFile();

    /// Template function for reading XML
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

    /// Template function for writing XML
    template<class T> void write_xml(const T& t)
    {
      open_file();
      typedef typename T::XMLHandler Handler;
      Handler::write(t, *outstream, 1);
      close_file();
    }

    // Input

    void operator>> (Mesh& input)                  { read_xml(input); }
    void operator>> (LocalMeshData& input)         { read_xml(input); }
    void operator>> (GenericMatrix& input)         { read_xml(input); }
    void operator>> (GenericVector& input)         { read_xml(input); }
    void operator>> (Parameters& input)            { read_xml(input); }
    void operator>> (FunctionPlotData&  input)     { read_xml(input); }
    void operator>> (MeshFunction<int>&  input)    { read_xml(input); }
    void operator>> (MeshFunction<uint>&  input)   { read_xml(input); }
    void operator>> (MeshFunction<double>&  input) { read_xml(input); }

    void operator>> (std::vector<int>& x)                             { read_xml_array(x); }
    void operator>> (std::vector<uint>& x)                            { read_xml_array(x); }
    void operator>> (std::vector<double>& x)                          { read_xml_array(x); }
    void operator>> (std::map<uint, int>& map)                        { read_xml_map(map); }
    void operator>> (std::map<uint, uint>& map)                       { read_xml_map(map); }
    void operator>> (std::map<uint, double>& map)                     { read_xml_map(map); }
    void operator>> (std::map<uint, std::vector<int> >& array_map)    { read_xml_map(array_map); }
    void operator>> (std::map<uint, std::vector<uint> >& array_map)   { read_xml_map(array_map); }
    void operator>> (std::map<uint, std::vector<double> >& array_map) { read_xml_map(array_map); }

    // Output

    void operator<< (const Mesh& output)                  { write_xml(output); }
    void operator<< (const GenericMatrix& output)         { write_xml(output); }
    void operator<< (const GenericVector& output)         { write_xml(output); }
    void operator<< (const Parameters& output)            { write_xml(output); }
    void operator<< (const FunctionPlotData& output)      { write_xml(output); }
    void operator<< (const MeshFunction<int>&  output)    { write_xml(output); }
    void operator<< (const MeshFunction<uint>&  output)   { write_xml(output); }
    void operator<< (const MeshFunction<double>&  output) { write_xml(output); }

    void operator<< (const std::vector<int>& x)                             { write_xml_array(x); }
    void operator<< (const std::vector<uint>& x)                            { write_xml_array(x); }
    void operator<< (const std::vector<double>& x)                          { write_xml_array(x); }
    void operator<< (const std::map<uint, int>& map)                        { write_xml_map(map); }
    void operator<< (const std::map<uint, uint>& map)                       { write_xml_map(map); }
    void operator<< (const std::map<uint, double>& map)                     { write_xml_map(map); }
    void operator<< (const std::map<uint, std::vector<int> >& array_map)    { write_xml_map(array_map); }
    void operator<< (const std::map<uint, std::vector<uint> >& array_map)   { write_xml_map(array_map); }
    void operator<< (const std::map<uint, std::vector<double> >& array_map) { write_xml_map(array_map); }

    // Friends
    friend void sax_start_element(void *ctx, const xmlChar *name, const xmlChar **attrs);
    friend void sax_end_element(void *ctx, const xmlChar *name);

    void validate(std::string filename);

    void write();

    void parse();

    void push(XMLHandler* handler);

    void pop();

    XMLHandler* top();

  private:

    template<class T> void read_xml_map(T& map)
    {
      info(TRACE, "Reading map from file %s.", filename.c_str());
      XMLMap xml_map(map, *this);
      XMLDolfin xml_dolfin(xml_map, *this);
      xml_dolfin.handle();
      parse();
      if ( !handlers.empty() )
        error("Hander stack not empty. Something is wrong!");
    }

    template<class T> void read_xml_array(T& x)
    {
      info(TRACE, "Reading array from file %s.", filename.c_str());
      XMLArray xml_array(x, *this);
      XMLDolfin xml_dolfin(xml_array, *this);
      xml_dolfin.handle();
      parse();
      if ( !handlers.empty() )
        error("Hander stack not empty. Something is wrong!");
    }

    template<class T> void write_xml_map(const T& map)
    {
      open_file();
      XMLMap::write(map, *outstream, 1);
      close_file();
    }

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
