// Copyright (C) 2007 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-11
// Last changed: 2009-03-11

#ifndef __NEWXMLFINITEELEMENT_H
#define __NEWXMLFINITEELEMENT_H

#include "XMLHandler.h"

namespace dolfin
{

  class NewXMLFiniteElement: public XMLHandler
  {
  public:

    NewXMLFiniteElement(std::string& signature, NewXMLFile& parser);
    
    void start_element(const xmlChar* name, const xmlChar** attrs);
    void end_element  (const xmlChar* name);
    
    static void write(const std::string& signature, std::ostream& outfile, uint indentation_level=0);
    
  private:
    
    enum parser_state { OUTSIDE, INSIDE_FINITE_ELEMENT, DONE };
    
    void read_finite_element(const xmlChar* name, const xmlChar** attrs);
    
    std::string& signature;
    parser_state state;
    
  };
  
}

#endif
