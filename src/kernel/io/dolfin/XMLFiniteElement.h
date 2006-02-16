// Copyright (C) 2006Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-17
// Last changed: 2006-02-17

#ifndef __XML_FINITE_ELEMENT_H
#define __XML_FINITE_ELEMENT_H

#include "XMLObject.h"

namespace dolfin
{

  class FiniteElement;
  
  class XMLFiniteElement : public XMLObject
  {
  public:

    XMLFiniteElement(FiniteElement& element);
    
    void startElement(const xmlChar* name, const xmlChar** attrs);
    void endElement  (const xmlChar* name);
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_FINITE_ELEMENT, DONE };
    
    void readFiniteElement(const xmlChar* name, const xmlChar** attrs);
    
    FiniteElement& element;
    ParserState state;
    
  };
  
}

#endif
