// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __XML_OBJECT_H
#define __XML_OBJECT_H

#include <parser.h>
#include <string>

#include <dolfin/constants.h>

namespace dolfin {

  class XMLObject {
  public:

	 XMLObject();
	 
	 virtual void startElement (const xmlChar *name, const xmlChar **attrs) = 0;
	 virtual void endElement   (const xmlChar *name) = 0;

	 // Write message before and after reading file
	 virtual void reading(string filename) {};
	 virtual void done() {};
	 
	 bool dataOK();
	 
  protected:

	 void parseIntegerRequired (const xmlChar *name, const xmlChar **attrs, const char *attribute, int *value);
	 void parseIntegerOptional (const xmlChar *name, const xmlChar **attrs, const char *attribute, int *value);
	 void parseRealRequired    (const xmlChar *name, const xmlChar **attrs, const char *attribute, real *value);
	 void parseRealOptional    (const xmlChar *name, const xmlChar **attrs, const char *name, real *value);

	 bool ok;
	 
  };
  
}

#endif
