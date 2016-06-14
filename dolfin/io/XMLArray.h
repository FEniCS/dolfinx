// Copyright (C) 2011 Garth N. Wells
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
// Modified by Anders Logg 2011
//
// First added:  2006-07-02
// Last changed: 2011-11-14

#ifndef __XMLARRAY_H
#define __XMLARRAY_H

#include <ostream>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include "dolfin/common/Array.h"
#include "dolfin/log/log.h"
#include "pugixml.hpp"

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  /// I/O of array data in XML format

  class XMLArray
  {
  public:

    /// Read XML vector. Vector must have correct size.
    template<typename T>
    static void read(std::vector<T>& x, const pugi::xml_node xml_dolfin);

    /// Write the XML file
    template<typename T>
    static void write(const std::vector<T>& x, const std::string type,
                      pugi::xml_node xml_node);

  };

  //---------------------------------------------------------------------------
  template<typename T>
  void XMLArray::read(std::vector<T>& x, const pugi::xml_node xml_node)
  {
    // Check that we have a XML Array
    const pugi::xml_node array = xml_node.child("array");
    if (!array)
    {
      dolfin_error("XMLArray.h",
                   "read array from XML file",
                   "Unable to find <array> tag in XML file");
    }

    // Get size and type
    const std::size_t size = array.attribute("size").as_uint();
    const std::string type  = array.attribute("type").value();
    if (type != "double")
    {
      dolfin_error("XMLArray.h",
                   "read array from XML file",
                   "XML I/O of Array objects only supported when the value type is 'double'");
    }

    // Iterate over array entries
    x.resize(size);
    Array<std::size_t> indices(size);
    for (pugi::xml_node_iterator it = array.begin(); it != array.end(); ++it)
    {
      const std::size_t index = it->attribute("index").as_uint();
      const double value = it->attribute("value").as_double();
      dolfin_assert(index < size);
      indices[index] = index;
      x[index] = value;
    }
  }
  //---------------------------------------------------------------------------
  template<typename T>
  void XMLArray::write(const std::vector<T>& x, const std::string type,
                       pugi::xml_node xml_node)
  {
    // Add array node
    pugi::xml_node array_node = xml_node.append_child("array");

    // Add attributes
    const std::size_t size = x.size();
    array_node.append_attribute("type") = type.c_str();
    array_node.append_attribute("size") = (unsigned int) size;

    // Add data
    for (std::size_t i = 0; i < size; ++i)
    {
      pugi::xml_node element_node = array_node.append_child("element");
      element_node.append_attribute("index") = (unsigned int) i;
      // NOTE: Casting to a string to avoid loss of precision when
      //       pugixml performs double-to-char conversion
      element_node.append_attribute("value")
        = boost::str(boost::format("%.15e") % x[i]).c_str();
    }
  }
  //---------------------------------------------------------------------------
}

#endif
