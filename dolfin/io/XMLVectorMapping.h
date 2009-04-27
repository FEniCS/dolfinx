// Copyright (C) 2009 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-02-26
// Last changed: 2009-02-26

#ifndef __NEW_XML_VECTORMAPPING_H
#define __NEW_XML_VECTORMAPPING_H

#include <map>
#include <vector>
#include "XMLObject.h"

namespace dolfin
{
  
  class XMLVectorMapping : public XMLObject
  {
  public:

    XMLVectorMapping(std::map<uint, std::vector<uint> >& mvec);
    ~XMLVectorMapping();
    
    void start_element (const xmlChar* name, const xmlChar** attrs);
    void end_element   (const xmlChar* name);
    
    void open(std::string filename);
    bool close();
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_VECTORMAPPING, INSIDE_MAP, INSIDE_VECTOR, DONE };
    enum VectorMappingType { INT, UINT, DOUBLE, BOOL, UNSET };
    
    void startVectorMapping(const xmlChar* name, const xmlChar** attrs);
    void read_entities    (const xmlChar* name, const xmlChar** attrs);

    ParserState state;
    VectorMappingType mvec_type;
    std::map<uint, std::vector<uint> >* _umvec;

  };
  
}

#endif
