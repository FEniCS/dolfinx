// Copyright (C) 2007 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-11
// Last changed: 2009-03-11

#include <dolfin/log/dolfin_log.h>
#include "NewXMLDofMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NewXMLDofMap::NewXMLDofMap(std::string& signature, NewXMLFile& parser)
  : XMLHandler(parser), signature(signature), state(OUTSIDE)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewXMLDofMap::start_element(const xmlChar* name, const xmlChar** attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "dofmap") == 0 )
    {
      read_dof_map(name, attrs);
      state = INSIDE_DOF_MAP;
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void NewXMLDofMap::end_element(const xmlChar* name)
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
void NewXMLDofMap::write(const std::string& signature, std::ofstream& outfile, uint indentation_level)
{
  outfile << std::setw(indentation_level) << "" << "<dofmap signature=\"" << signature << "\"/>" << std::endl;
}
//-----------------------------------------------------------------------------
void NewXMLDofMap::read_dof_map(const xmlChar* name, const xmlChar** attrs)
{
  // Parse values
  signature = parse_string(name, attrs, "signature");
}
//-----------------------------------------------------------------------------
