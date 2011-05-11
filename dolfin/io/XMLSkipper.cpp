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
