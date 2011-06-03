// Copyright (C) 2009 Ola Skavhaug
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
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
