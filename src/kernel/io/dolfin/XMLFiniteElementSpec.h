// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-19
// Last changed: 2006-05-23

#ifndef __XML_FINITE_ELEMENT_SPEC_H
#define __XML_FINITE_ELEMENT_SPEC_H

#include "XMLObject.h"

namespace dolfin
{

  class FiniteElementSpec;
  
  class XMLFiniteElementSpec : public XMLObject
  {
  public:

    XMLFiniteElementSpec(FiniteElementSpec& spec);
    
    void startElement(const xmlChar* name, const xmlChar** attrs);
    void endElement  (const xmlChar* name);
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_FINITE_ELEMENT, DONE };
    
    void readFiniteElementSpec(const xmlChar* name, const xmlChar** attrs);
    
    FiniteElementSpec& spec;
    ParserState state;
    
  };
  
}

#endif
