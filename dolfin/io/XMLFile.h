// Copyright (C) 2009 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-03
// Last changed: 2009-04-01

#ifndef __XMLFILE_H
#define __XMLFILE_H

#include <fstream>
#include <stack>
#include <string>
#include <map>
#include <vector>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/graph/Graph.h>
#include <dolfin/plot/FunctionPlotData.h>
#include <dolfin/parameter/ParameterList.h>
#include <libxml/parser.h>
#include "GenericFile.h"
#include "XMLMesh.h"
#include "XMLMeshFunction.h"
#include "XMLGraph.h"
#include "XMLMatrix.h"
#include "XMLLocalMeshData.h"
#include "XMLParameterList.h"
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

    // Input

    template<class T> void read_xml(T& t)
    {
      typedef typename T::XMLHandler Handler;
      Handler xml_handler(t, *this);
      XMLDolfin xml_dolfin(xml_handler, *this);
      xml_dolfin.handle();
      parse();
      if ( !handlers.empty() )
        error("Handler stack not empty. Something is wrong!");
    }

    template<class T> void write_xml(const T& t)
    {
      open_file();
      typedef typename T::XMLHandler Handler;
      Handler::write(t, *outstream, 1);
      close_file();
    }

    // Input

    void operator>> (Mesh& input)          { read_xml(input); }
    void operator>> (LocalMeshData& input) { read_xml(input); }
    void operator>> (Graph&  input)        { read_xml(input); }
    void operator>> (GenericMatrix&  input){ read_xml(input); }
    void operator>> (GenericVector&  input){ read_xml(input); }
    void operator>> (ParameterList&  input){ read_xml(input); }
    void operator>> (FunctionPlotData&  input){ read_xml(input); }
    void operator>> (MeshFunction<int>&  input){ read_xml(input); }
    void operator>> (MeshFunction<uint>&  input){ read_xml(input); }
    void operator>> (MeshFunction<double>&  input){ read_xml(input); }

    void operator>> (std::vector<int> & x);
    void operator>> (std::vector<uint> & x);
    void operator>> (std::vector<double> & x);
    void operator>> (std::map<uint, int>& map);
    void operator>> (std::map<uint, uint>& map);
    void operator>> (std::map<uint, double>& map);
    void operator>> (std::map<uint, std::vector<int> >& array_map);
    void operator>> (std::map<uint, std::vector<uint> >& array_map);
    void operator>> (std::map<uint, std::vector<double> >& array_map);

    // Output

    void operator<< (const Mesh& output)         { write_xml(output); }
    void operator<< (const Graph& output)         { write_xml(output); }
    void operator<< (const GenericMatrix& output) { write_xml(output); }
    void operator<< (const GenericVector& output) { write_xml(output); }
    void operator<< (const ParameterList& output) { write_xml(output); }
    void operator<< (const FunctionPlotData& output) { write_xml(output); }
    void operator<< (const MeshFunction<int>&  output){ write_xml(output); }
    void operator<< (const MeshFunction<uint>&  output){ write_xml(output); }
    void operator<< (const MeshFunction<double>&  output){ write_xml(output); }

    void operator<< (const std::vector<int> & x);
    void operator<< (const std::vector<uint> & x);
    void operator<< (const std::vector<double> & x);
    void operator<< (const std::map<uint, int>& map);
    void operator<< (const std::map<uint, uint>& map);
    void operator<< (const std::map<uint, double>& map);
    void operator<< (const std::map<uint, std::vector<int> >& array_map);
    void operator<< (const std::map<uint, std::vector<uint> >& array_map);
    void operator<< (const std::map<uint, std::vector<double> >& array_map);

    // Friends
    friend void new_sax_start_element (void *ctx, const xmlChar *name, const xmlChar **attrs);
    friend void new_sax_end_element   (void *ctx, const xmlChar *name);

    void validate(std::string filename);

    void write() {}

    void parse();

    void push(XMLHandler* handler);

    void pop();

    XMLHandler* top();

  private:
    std::stack<XMLHandler*> handlers;
    xmlSAXHandler* sax;
    std::ostream* outstream;

    void start_element(const xmlChar *name, const xmlChar **attrs);
    void end_element  (const xmlChar *name);

    void open_file();
    void close_file();

  };

  // Callback functions for the SAX interface

  void new_sax_start_document (void *ctx);
  void new_sax_end_document   (void *ctx);
  void new_sax_start_element  (void *ctx, const xmlChar *name, const xmlChar **attrs);
  void new_sax_end_element    (void *ctx, const xmlChar *name);

  void new_sax_warning     (void *ctx, const char *msg, ...);
  void new_sax_error       (void *ctx, const char *msg, ...);
  void new_sax_fatal_error (void *ctx, const char *msg, ...);

  // Callback functions for Relax-NG Schema
  void new_rng_parser_error(void *user_data, xmlErrorPtr error);
  void new_rng_valid_error (void *user_data, xmlErrorPtr error);

}
#endif
