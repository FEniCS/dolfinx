// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-13
// Last changed: 2007-04-13

#ifndef __XML_DOF_MAP_H
#define __XML_DOF_MAP_H

#include "XMLObject.h"

namespace dolfin
{

  class XMLDofMap: public XMLObject
  {
  public:

    XMLDofMap(std::string& signature);
    
    void startElement(const xmlChar* name, const xmlChar** attrs);
    void endElement  (const xmlChar* name);
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_DOF_MAP, DONE };
    
    void readDofMap(const xmlChar* name, const xmlChar** attrs);
    
    std::string& signature;
    ParserState state;
    
  };
  
}

#endif
