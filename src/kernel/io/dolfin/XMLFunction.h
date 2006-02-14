// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-07-15
// Last changed: 2005

#ifndef __XML_FUNCTION_H
#define __XML_FUNCTION_H

#include "XMLObject.h"

namespace dolfin {

  class Function;
  
  class XMLFunction : public XMLObject {
  public:

	 XMLFunction(Function& f);

	 void startElement (const xmlChar *name, const xmlChar **attrs);
	 void endElement   (const xmlChar *name);
	 
  private:
	 
	 enum ParserState { OUTSIDE, INSIDE_FUNCTION, DONE };
	 
	 void readFunction  (const xmlChar *name, const xmlChar **attrs);
	 
	 Function& f;
	 ParserState state;
	 
  };
  
}

#endif
