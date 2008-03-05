// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-13
// Last changed: 2007-04-13

#include <dolfin/log/dolfin_log.h>
#include "XMLDofMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLDofMap::XMLDofMap(std::string& signature)
  : XMLObject(), signature(signature)
{
  state = OUTSIDE;
}
//-----------------------------------------------------------------------------
void XMLDofMap::startElement(const xmlChar* name, const xmlChar** attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "dofmap") == 0 )
    {
      readDofMap(name, attrs);
      state = INSIDE_DOF_MAP;
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLDofMap::endElement(const xmlChar* name)
{
  switch ( state )
  {
  case INSIDE_DOF_MAP:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "dofmap") == 0 )
    {
      state = DONE;
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLDofMap::readDofMap(const xmlChar* name,
                                         const xmlChar** attrs)
{
  // Parse values
  signature = parseString(name, attrs, "signature");
}
//-----------------------------------------------------------------------------
