// Copyright (C) 2009 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-02-26
// Last changed: 2009-02-26

#include "XMLVectorMapping.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLVectorMapping::XMLVectorMapping(std::map<uint, std::vector<uint> >& mvec)
: XMLObject(), 
  state(OUTSIDE),
  mvec_type(UNSET),
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
void XMLVectorMapping::startElement(const xmlChar* name, const xmlChar** attrs)
{
  switch ( state )
  {
  case OUTSIDE:

    if ( xmlStrcasecmp(name, (xmlChar *) "vectormapping") == 0 )
    {
      startVectorMapping(name, attrs);
      state = INSIDE_VECTORMAPPING;
    }
    break;
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLVectorMapping::startVectorMapping(const xmlChar *name, const xmlChar **attrs)
{
}
//-----------------------------------------------------------------------------
void XMLVectorMapping::endElement(const xmlChar* name)
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
void XMLVectorMapping::readEntities(const xmlChar* name, const xmlChar** attrs)
{
}
//-----------------------------------------------------------------------------
