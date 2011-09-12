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
// First added:  2006-07-02
// Last changed:

#ifndef __XMLARRAY_H
#define __XMLARRAY_H

#include <ostream>
#include <string>
#include <boost/lexical_cast.hpp>
#include "dolfin/common/Array.h"
#include "dolfin/log/log.h"
#include "pugixml.hpp"

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  class XMLArray
  {
  public:

    // Read XML vector. Vector must have correct size.
    template<class T>
    static void read(Array<T>& x, const pugi::xml_node xml_dolfin);

    /// Write the XML file
    template<class T>
    static void write(const Array<T>& x, const std::string type,
                      pugi::xml_node xml_node);

  };

  //-----------------------------------------------------------------------------
  template<class T>
  void XMLArray::read(Array<T>& x, const pugi::xml_node xml_node)
  {
    // Check that we have a XML Array
    const pugi::xml_node array = xml_node.child("array");
    if (!array)
      error("XMLVector::read: not a DOLFIN array inside Vector XML file.");

    // Get size and type
    const unsigned int size = array.attribute("size").as_uint();
    const std::string type  = array.attribute("type").value();
    if (type != "double")
      error("XMLArray::read only supports type double.");

    // Iterate over array entries
    x.resize(size);
    Array<unsigned int> indices(size);
    for (pugi::xml_node_iterator it = array.begin(); it != array.end(); ++it)
    {
      const unsigned int index = it->attribute("index").as_uint();
      const double value = it->attribute("value").as_double();
      assert(index < size);
      indices[index] = index;
      x[index] = value;
    }
  }
  //-----------------------------------------------------------------------------
  template<class T>
  void XMLArray::write(const Array<T>& x, const std::string type,
                       pugi::xml_node xml_node)
  {
    // Add array node
    pugi::xml_node array_node = xml_node.append_child("array");

    // Add attributes
    const unsigned int size = x.size();
    array_node.append_attribute("type") = type.c_str();
    array_node.append_attribute("size") = size;

    // Add data
    for (uint i = 0; i < size; ++i)
    {
      pugi::xml_node element_node = array_node.append_child("element");
      element_node.append_attribute("index") = i;
      // NOTE: Casting to a string to avoid loss of precision when
      //       pugixml performs double-to-char conversion
      element_node.append_attribute("value") = boost::lexical_cast<std::string>(x[i]).c_str();
    }
  }
  //-----------------------------------------------------------------------------
}

#endif
