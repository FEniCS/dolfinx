// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __XML_VECTOR_H
#define __XML_VECTOR_H

#include "XMLObject.h"

namespace dolfin {

  class Vector;
  
  class XMLVector : public XMLObject {
  public:

	 XMLVector(Vector& vector);

	 void startElement (const xmlChar *name, const xmlChar **attrs);
	 void endElement   (const xmlChar *name);
	 
  private:
	 
	 enum ParserState { OUTSIDE, INSIDE_VECTOR, DONE };
	 
	 void readVector  (const xmlChar *name, const xmlChar **attrs);
	 void readElement (const xmlChar *name, const xmlChar **attrs);
	 
	 Vector& x;
	 ParserState state;
	 
  };
  
}

#endif
