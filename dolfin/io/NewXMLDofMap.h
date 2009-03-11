// Copyright (C) 2007 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-11
// Last changed: 2009-03-11

#ifndef __NEWXMLDOFMAP_H
#define __NEWXMLDOFMAP_H

#include "XMLHandler.h"

namespace dolfin
{

  class NewXMLDofMap: public XMLHandler
  {
  public:

    NewXMLDofMap(std::string& signature, NewXMLFile& parser);
    
    void start_element(const xmlChar* name, const xmlChar** attrs);
    void end_element  (const xmlChar* name);
    
    static void write(const std::string& signature, std::ofstream& outfile, uint indentation_level=0);
    
  private:
    
    enum parser_state { OUTSIDE, INSIDE_DOF_MAP, DONE };
    
    void read_dof_map(const xmlChar* name, const xmlChar** attrs);
    
    std::string& signature;
    parser_state state;
    
  };
  
}

#endif
