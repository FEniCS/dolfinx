// Copyright (C) 2009 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-08-06
// Last changed: 2009-08-06

#include "XMLSkipper.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLSkipper::XMLSkipper(std::string name, XMLFile& parser) 
  : XMLHandler(parser), name(name) 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLSkipper::end_element(const xmlChar* name)
{
  if (this->name.compare((const char*)name) == 0)
    release();
}
//-----------------------------------------------------------------------------
