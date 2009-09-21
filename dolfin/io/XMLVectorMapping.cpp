// Copyright (C) 2009 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
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
