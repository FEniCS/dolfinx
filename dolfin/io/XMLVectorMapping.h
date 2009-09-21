// Copyright (C) 2009 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-02-26
// Last changed: 2009-09-09

#ifndef __XML_VECTORMAPPING_H
#define __XML_VECTORMAPPING_H

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

    static void write(const std::map<uint, std::vector<uint> >& amap, std::ostream& outfile,uint indentation_level=0);

  private:

    enum ParserState { OUTSIDE, INSIDE_VECTORMAPPING, INSIDE_MAP, INSIDE_VECTOR, DONE };

    void start_vector_mapping(const xmlChar* name, const xmlChar** attrs);
    void read_entities    (const xmlChar* name, const xmlChar** attrs);

    ParserState state;
    std::map<uint, std::vector<uint> >* _umvec;

  };

}

#endif
