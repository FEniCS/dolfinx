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
// Last changed: 2011-09-15

#ifndef __XML_UTILS_H
#define __XML_UTILS_H

#include <string>

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  namespace xmlutils
  {
    // This file provides a small number of utility functions that may be
    // useful when parsing XML using pugixml.

    /// Get XML node with given name, either the given node itself or a
    /// child node. An error message is thrown if node is not found.
    const pugi::xml_node get_node(const pugi::xml_node& xml_node,
                                  std::string node_name);

    // Check that xml_node has name 'name'. If not, throw error.
    void check_node_name(const pugi::xml_node& xml_node,
                         const std::string name);

    // Check that xml_node has attribute 'name'. If not, throw error.
    void check_has_attribute(const pugi::xml_node& xml_node,
                             const std::string name);
  }

}

#endif
