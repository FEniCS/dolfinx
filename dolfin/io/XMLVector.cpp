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
// Last changed: 2011-11-17

#include <iomanip>
#include <iostream>
#include <string>
#include "pugixml.hpp"

#include "dolfin/common/Array.h"
#include "dolfin/common/MPI.h"
#include "dolfin/fem/GenericDofMap.h"
#include "dolfin/function/FunctionSpace.h"
#include "dolfin/la/GenericVector.h"
#include "dolfin/mesh/Mesh.h"
#include "XMLArray.h"
#include "XMLVector.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLVector::read(GenericVector& x, const pugi::xml_node xml_dolfin)
{
  // Read data in to vectors
  std::vector<double> data;
  std::vector<dolfin::la_index> indices;
  read(data, indices, xml_dolfin);

  // Set data (GenericVector::apply will be called by calling function)
  x.set(data.data(), data.size(), indices.data());
}
//-----------------------------------------------------------------------------
void XMLVector::read(std::vector<double>& x,
                     std::vector<dolfin::la_index>& indices,
                     const pugi::xml_node xml_dolfin)
{
  // Check that we have a XML Vector
  const pugi::xml_node xml_vector_node = xml_dolfin.child("vector");
  if (!xml_vector_node)
  {
    dolfin_error("XMLVector.cpp",
                 "read vector from XML file",
                 "Not a DOLFIN Vector XML file");
  }

  // Get type and size
  const pugi::xml_node array = xml_vector_node.child("array");
  if (!array)
  {
    dolfin_error("XMLVector.cpp",
                 "read vector from XML file",
                 "Expecting an Array inside a DOLFIN Vector XML file");
  }

  const std::size_t size = array.attribute("size").as_uint();
  const std::string type = array.attribute("type").value();

  // Check if size is zero
  if (size == 0)
  {
    dolfin_error("XMLVector.cpp",
                 "read vector from XML file",
                 "size is zero");
  }

  // Iterate over array entries
  x.resize(size);
  indices.resize(size);
  for (pugi::xml_node_iterator it = array.begin(); it != array.end(); ++it)
  {
    const std::size_t index = it->attribute("index").as_uint();
    const double value = it->attribute("value").as_double();
    dolfin_assert(index < size);
    indices[index] = index;
    x[index] = value;
  }
}
//-----------------------------------------------------------------------------
std::size_t XMLVector::read_size(const pugi::xml_node xml_dolfin)
{
  // Check that we have a XML Vector
  const pugi::xml_node xml_vector_node = xml_dolfin.child("vector");
  if (!xml_vector_node)
  {
    dolfin_error("XMLVector.cpp",
                 "read vector from XML file",
                 "Not a DOLFIN Vector XML file");
  }

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
  std::vector<double> x;
  if (MPI::size(vector.mpi_comm()) > 1)
    vector.gather_on_zero(x);
  else
    vector.get_local(x);

  const std::size_t size = vector.size();

  // Check if size is zero
  if (size == 0)
  {
    dolfin_error("XMLVector.cpp",
                 "write vector to XML file",
                 "size is zero");
  }

  if (write_to_stream)
  {
    dolfin_assert(size == x.size());

    // Add vector node
    pugi::xml_node vector_node = xml_node.append_child("vector");

    // Write data
    XMLArray::write(x, "double", vector_node);
  }
}
//-----------------------------------------------------------------------------
