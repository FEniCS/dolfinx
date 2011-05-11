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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-02-26
// Last changed: 2009-09-09

#include "XMLIndent.h"
#include "XMLVectorMapping.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLVectorMapping::XMLVectorMapping(std::map<uint, std::vector<uint> >& mvec)
: XMLObject(),
  state(OUTSIDE),
  _umvec(&mvec)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLVectorMapping::~XMLVectorMapping()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLVectorMapping::start_element(const xmlChar* name, const xmlChar** attrs)
{
  switch ( state )
  {
  case OUTSIDE:

    if ( xmlStrcasecmp(name, (xmlChar *) "vectormapping") == 0 )
    {
      start_vector_mapping(name, attrs);
      state = INSIDE_VECTORMAPPING;
    }
    break;

  case INSIDE_VECTORMAPPING:
    if ( xmlStrcasecmp(name, (xmlChar *) "entity") == 0 )
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLVectorMapping::start_vector_mapping(const xmlChar *name, const xmlChar **attrs)
{
}
//-----------------------------------------------------------------------------
void XMLVectorMapping::end_element(const xmlChar* name)
{
}
//-----------------------------------------------------------------------------
void XMLVectorMapping::open(std::string filename)
{
}
//-----------------------------------------------------------------------------
bool XMLVectorMapping::close()
{
  return true;
}
//-----------------------------------------------------------------------------
void XMLVectorMapping::write(const std::map<uint, std::vector<uint> >& amap, std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<array_map key_type=\"uint\" value_type=\"uint\">" << std::endl;
  ++indent;
  --indent;
  outfile << indent() << "</array_map>" << std::endl;
}

//-----------------------------------------------------------------------------
void XMLVectorMapping::read_entities(const xmlChar* name, const xmlChar** attrs)
{
}
//-----------------------------------------------------------------------------
