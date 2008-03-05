// Copyright (C) 2006 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-15
// Last changed: 2006-05-23

#ifndef __XML_FUNCTION_H
#define __XML_FUNCTION_H

#include "XMLObject.h"

namespace dolfin
{

  class Function;
  
  class XMLFunction : public XMLObject
  {
  public:

    XMLFunction(Function& f);
    
    void startElement(const xmlChar* name, const xmlChar** attrs);
    void endElement  (const xmlChar* name);
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_FUNCTION, DONE };
    
    void readFunction  (const xmlChar* name, const xmlChar** attrs);
    
    Function& f;
    ParserState state;
    
  };
  
}

#endif
