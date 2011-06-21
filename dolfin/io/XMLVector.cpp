// Copyright (C) 2002-2011 Anders Logg, Ola Skavhaug and Garth N. Wells
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
// Last changed: 2006-10-16

#include <iomanip>
#include <iostream>
#include "pugixml.hpp"

#include "dolfin/common/Array.h"
#include <dolfin/common/MPI.h>
#include "dolfin/la/GenericVector.h"
#include "XMLIndent.h"
#include "XMLVector.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLVector::read(GenericVector& x, const pugi::xml_node xml_dolfin)
{
  // Check that we have a XML Vector
  const pugi::xml_node xml_vector_node = xml_dolfin.child("vector");
  if (!xml_vector_node)
    error("Not a DOLFIN Vector file.");

  // Get type and size
  const pugi::xml_node array = xml_vector_node.child("array");
  if (!array)
    std::cout << "Not a DOLFIN Array" << std::endl;

  const unsigned int size = array.attribute("size").as_uint();
  const std::string type  = array.attribute("type").value();

  Array<double> data(size);
  Array<unsigned int> indices(size);

  // Iterate over array entries
  for (pugi::xml_node_iterator it = array.begin(); it != array.end(); ++it)
  {
    const unsigned int index = it->attribute("index").as_uint();
    const double value = it->attribute("value").as_double();
    assert(index < size);
    indices[index] = index;
    data[index] = value;
  }

  // Resize vector and add data
  x.resize(size);
  x.set(data.data().get(), size, indices.data().get());
  x.apply("insert");
}
//-----------------------------------------------------------------------------
void XMLVector::write(const GenericVector& vector, std::ostream& outfile,
                      unsigned int indentation_level)
{
  bool write_to_stream = false;
  if (MPI::process_number() == 0)
    write_to_stream = true;

  XMLIndent indent(indentation_level);

  // Write vector header
  if (write_to_stream)
  {
    outfile << indent() << "<vector>" << std::endl;
    ++indent;

    // Write array header
    outfile << indent() << "<array type=\"double\" size=\"" << vector.size()
      << "\">" << std::endl;
    ++indent;
  }

  // Gather entries from process i on process 0
  Array<double> x;
  if (MPI::num_processes() >1)
    vector.gather_on_zero(x);
  else
    vector.get_local(x);

  // Write vector entries
  if (write_to_stream)
  {
    for (uint i = 0; i < x.size(); ++i)
    {
      outfile << indent()
        << "<element index=\"" << i
        << "\" value=\"" << std::setprecision(16) << x[i]
        << "\"/>" << std::endl;
    }

    --indent;
    outfile << indent() << "</array>" << std::endl;
    --indent;

    // Write vector footer
    if (write_to_stream)
      outfile << indent() << "</vector>" << std::endl;
  }
}
//-----------------------------------------------------------------------------
