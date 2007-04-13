// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-13
// Last changed: 2007-04-13

#ifndef __XML_FINITE_ELEMENT_H
#define __XML_FINITE_ELEMENT_H

#include "XMLObject.h"

namespace dolfin
{

  class XMLFiniteElement: public XMLObject
  {
  public:

    XMLFiniteElement(std::string& signature);
    
    void startElement(const xmlChar* name, const xmlChar** attrs);
    void endElement  (const xmlChar* name);
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_FINITE_ELEMENT, DONE };
    
    void readFiniteElement(const xmlChar* name, const xmlChar** attrs);
    
    std::string& signature;
    ParserState state;
    
  };
  
}

#endif
