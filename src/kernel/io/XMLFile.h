// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __XML_FILE_H
#define __XML_FILE_H

#include <parser.h>

#include <dolfin/dolfin_constants.h>
#include "GenericFile.h"

namespace dolfin {

  class Vector;
  class XMLObject;
  
  enum ParserState { OUTSIDE, VECTOR_OUTSIDE, VECTOR_INSIDE, DONE };
  
  class XMLFile : public GenericFile {
  public:

	 XMLFile(const std::string filename);
	 ~XMLFile();
	 
	 // Input
	 
	 void operator>> (Vector &vector);
	 void operator>> (SparseMatrix &sparseMatrix);
	 void operator>> (Grid &grid);
	 
	 // Output
	 
	 void operator<< (const Vector &vector);
	 void operator<< (const SparseMatrix &sparseMatrix);
	 void operator<< (const Grid &grid);

	 // Friends
	 
	 friend void sax_start_element (void *ctx, const xmlChar *name, const xmlChar **attrs);
	 friend void sax_end_element   (void *ctx, const xmlChar *name);

  private:

	 void parseFile();

	 // Parser and state
	 xmlSAXHandler sax;
	 ParserState state;

	 // Data
	 XMLObject *xmlObject;
	 
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
