// Copyright (C) 2009 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-03
// Last changed: 2009-03-04

#ifndef __NEWXMLFILE_H
#define __NEWXMLFILE_H

#include <fstream>
#include <stack>
#include <string>
#include <map>
#include <vector>
#include <libxml/parser.h>
#include "GenericFile.h"
#include "XMLHandler.h"

namespace dolfin
{

  class NewXMLFile: public GenericFile
  {
  public:

    /// Constructor
    NewXMLFile(const std::string filename, bool gzip);

    /// Destructor
    ~NewXMLFile();

    // Input

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

    void parse();

    void push(XMLHandler* handler);

    void pop();

    XMLHandler* top();

  private:
    std::stack<XMLHandler*> handlers;
    xmlSAXHandler* sax;
    std::ofstream outfile;

    void start_element(const xmlChar *name, const xmlChar **attrs);
    void end_element  (const xmlChar *name);

    void open_file();
    void close_file() { outfile.close(); }

  };

  // Callback functions for the SAX interface
  
  void new_sax_start_document (void *ctx);
  void new_sax_end_document   (void *ctx);
  void new_sax_start_element  (void *ctx, const xmlChar *name, const xmlChar **attrs);
  void new_sax_end_element    (void *ctx, const xmlChar *name);

  void new_sax_warning     (void *ctx, const char *msg, ...);
  void new_sax_error       (void *ctx, const char *msg, ...);
  void new_sax_fatal_error (void *ctx, const char *msg, ...);
 
}
#endif
