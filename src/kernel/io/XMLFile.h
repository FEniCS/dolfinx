// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __XML_FILE_H
#define __XML_FILE_H

#include <parser.h>

#include <dolfin/dolfin_constants.h>
#include "GenericFile.h"

namespace dolfin {

  class Vector;
  
  class XMLFile : public GenericFile {
  public:

	 XMLFile(const std::string filename);
	 ~XMLFile();
	 
	 // Input
	 
	 void operator>> (Vector &vector);
	 
	 // Output
	 
	 void operator<< (const Vector &vector);

	 // Friends
	 
	 friend void sax_start_element(void *ctx, const xmlChar *name, const xmlChar **attrs);

  protected:

	 void VectorInit  (const xmlChar *name, const xmlChar **attrs);
	 void VectorValue (const xmlChar *name, const xmlChar **attrs);
	 
  private:

	 void resetData();
	 void parseFile();
	 void parseIntegerRequired (const xmlChar *name, const xmlChar **attrs, const char *attribute, int *value);
	 void parseIntegerOptional (const xmlChar *name, const xmlChar **attrs, const char *attribute, int *value);
	 void parseRealRequired    (const xmlChar *name, const xmlChar **attrs, const char *attribute, real *value);
	 void parseRealOptional    (const xmlChar *name, const xmlChar **attrs, const char *name, real *value);

	 xmlSAXHandler sax;

	 // Data
	 Vector *vector;
	 
  };

  // Callback functions for the SAX interface
  
  void sax_start_document (void *ctx);
  void sax_end_document   (void *ctx);
  void sax_start_element  (void *ctx, const xmlChar *name, const xmlChar **attrs);
  void sax_end_element    (void *ctx, const xmlChar *name);

  static void sax_warning     (void *ctx, const char *msg, ...);
  static void sax_error       (void *ctx, const char *msg, ...);
  static void sax_fatal_error (void *ctx, const char *msg, ...);
  
}

#endif
