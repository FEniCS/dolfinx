// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __XML_FILE_H
#define __XML_FILE_H

#include <libxml/parser.h>

#include <dolfin/constants.h>
#include <dolfin/GenericFile.h>

namespace dolfin {
  
  class Vector;
  class Matrix;
  class Mesh;
  class Function;
  class XMLObject;
  
  class XMLFile : public GenericFile {
  public:
    
    XMLFile(const std::string filename);
    ~XMLFile();
    
    // Input
    
    void operator>> (Vector& x);
    void operator>> (Matrix& A);
    void operator>> (Mesh& mesh);
    
    // Output
    
    void operator<< (Vector& x);
    void operator<< (Matrix& A);
    
    // Friends
    
    friend void sax_start_element (void *ctx, const xmlChar *name, const xmlChar **attrs);
    friend void sax_end_element   (void *ctx, const xmlChar *name);
    
  private:
    
    void parseFile();
    void parseSAX();
    
    // Data
    XMLObject* xmlObject;
    
  };
  
  // Callback functions for the SAX interface
  
  void sax_start_document (void *ctx);
  void sax_end_document   (void *ctx);
  void sax_start_element  (void *ctx, const xmlChar *name, const xmlChar **attrs);
  void sax_end_element    (void *ctx, const xmlChar *name);

  void sax_warning     (void *ctx, const char *msg, ...);
  void sax_error       (void *ctx, const char *msg, ...);
  void sax_fatal_error (void *ctx, const char *msg, ...);
  
}

#endif
