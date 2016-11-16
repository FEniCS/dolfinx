// Copyright (C) 2011 Anders Logg
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
// First added:  2011-09-15
// Last changed: 2011-11-15

#include "pugixml.hpp"
#include <dolfin/log/log.h>
#include "xmlutils.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
const pugi::xml_node xmlutils::get_node(const pugi::xml_node& xml_node,
                                        std::string node_name)
{
  // Check node itself
  if (xml_node.name() == node_name)
    return xml_node;

  // Check child
  const pugi::xml_node child_node = xml_node.child(node_name.c_str());
  if (!child_node)
  {
    dolfin_error("xmlutils.cpp",
                 "read DOLFIN XML data",
                 "Unable to find tag <%s>", node_name.c_str());
  }

  return child_node;
}
//-----------------------------------------------------------------------------
void xmlutils::check_node_name(const pugi::xml_node& xml_node,
                               const std::string name)
{
  if (xml_node.name() != name)
  {
    dolfin_error("xmlutils.cpp",
                 "checking XML node name",
                 "Node name is \"%s\", expecting \"%s\"",
                  xml_node.name(), name.c_str());

  }
}
//-----------------------------------------------------------------------------
void xmlutils::check_has_attribute(const pugi::xml_node& xml_node,
                                   const std::string name)
{
  const pugi::xml_attribute attr = xml_node.attribute("xml_node");
  if (!attr)
  {
    dolfin_error("xmlutils.cpp",
                 "checking that XML node has attribute",
                 "Node  \"%s\" does not have expected attribute \"%s\"",
                  xml_node.name(), name.c_str());

  }
}
//-----------------------------------------------------------------------------
