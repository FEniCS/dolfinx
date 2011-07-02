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
// First added:  2002-12-06
// Last changed: 2011-06-30

#include <iomanip>
#include <iostream>
#include <string>
#include "pugixml.hpp"

#include "dolfin/common/Array.h"
#include "dolfin/common/MPI.h"
#include "dolfin/la/GenericVector.h"
#include "XMLArray.h"
#include "XMLVector.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLVector::read(GenericVector& x, const pugi::xml_node xml_dolfin)
{
  // Check that we have a XML Vector
  const pugi::xml_node xml_vector_node = xml_dolfin.child("vector");
  if (!xml_vector_node)
    error("XMLVector::read: not a DOLFIN Vector file.");

  // Get type and size
  const pugi::xml_node array = xml_vector_node.child("array");
  if (!array)
    error("XMLVector::read: not a DOLFIN array inside Vector XML file.");

  const unsigned int size = array.attribute("size").as_uint();
  const std::string type  = array.attribute("type").value();

  // Iterate over array entries
  Array<double> data(size);
  Array<unsigned int> indices(size);
  for (pugi::xml_node_iterator it = array.begin(); it != array.end(); ++it)
  {
    const unsigned int index = it->attribute("index").as_uint();
    const double value = it->attribute("value").as_double();
    assert(index < size);
    indices[index] = index;
    data[index] = value;
  }

  // Set data (GenericVector::apply will be called by calling function)
  x.set(data.data().get(), size, indices.data().get());
}
//-----------------------------------------------------------------------------
dolfin::uint XMLVector::read_size(const pugi::xml_node xml_dolfin)
{
  // Check that we have a XML Vector
  const pugi::xml_node xml_vector_node = xml_dolfin.child("vector");
  if (!xml_vector_node)
    error("XMLVector::read_size: not a DOLFIN Vector file.");

  // Get size
  const pugi::xml_node array = xml_vector_node.child("array");
  if (!array)
    std::cout << "XMLVector::read_size: not a DOLFIN Array" << std::endl;

  return array.attribute("size").as_uint();
}
//-----------------------------------------------------------------------------
void XMLVector::write(const GenericVector& vector, pugi::xml_node xml_node,
                      bool write_to_stream)
{
  // Gather entries from process i on process 0
  Array<double> x;
  if (MPI::num_processes() > 1)
    vector.gather_on_zero(x);
  else
    vector.get_local(x);

  const uint size = vector.size();

  if (write_to_stream)
  {
    assert(size == x.size());

    // Add vector node
    pugi::xml_node vector_node = xml_node.append_child("vector");

    // Write data
    XMLArray::write(x, "double", vector_node);
  }
}
//-----------------------------------------------------------------------------
