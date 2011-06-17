// Copyright (C) 2002-2006 Anders Logg and Ola Skavhaug
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
// Modified by Garth N. Wells, 2009.
//
// First added:  2009-03-06
// Last changed: 2009-06-15


#include <dolfin/common/Array.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/Vector.h>
#include <dolfin/common/MPI.h>
#include "XMLIndent.h"
#include "XMLVector.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLVector::XMLVector(GenericVector& vector, XMLFile& parser)
  : XMLHandler(parser), x(vector), state(OUTSIDE)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLVector::~XMLVector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLVector::start_element(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    if ( xmlStrcasecmp(name, (xmlChar *) "vector") == 0 )
      read_vector_tag(name, attrs);
    break;
  case INSIDE_VECTOR:
    if ( xmlStrcasecmp(name, (xmlChar *) "array") == 0 )
      read_array_tag(name, attrs);
    break;
  default:
    break;
  }
}
//-----------------------------------------------------------------------------
void XMLVector::end_element(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_VECTOR:
    if ( xmlStrcasecmp(name, (xmlChar *) "vector") == 0 )
    {
      end_vector();
      state = DONE;
      release();
    }
    break;
  default:
    break;
  }
}
//-----------------------------------------------------------------------------
void XMLVector::write(const GenericVector& vector, std::ostream& outfile,
                      uint indentation_level)
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
  vector.gather_on_zero(x);

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
  /*
  // Gather entries from process i on process 0 and write to file
  for (uint process  = 0; process < MPI::num_processes(); ++process)
  {

    // FIXME: Use more elgant approach. The gather functions are collective,
    //        i.e. must be called on all processes even though we only want to
    //        gather on process 0.
    uint n0 = 0;
    uint local_size = 0;
    if (MPI::process_number() == 0)
    {
      // Get (approximate) range for portion of vector on process
      const std::pair<uint, uint> range = MPI::local_range(process, vector.size());

      // Get offset anf compute local size
      n0 = range.first;
      local_size = range.second - range.first;
    }

    Array<uint> indices(local_size);
    for (uint i = 0; i < local_size; ++i)
      indices[i] = n0 + i;

    // Create local Array and gather values into it
    Array<double> vector_values;
    vector.gather(vector_values, indices);

    // Write vector entries
    if (write_to_stream)
    {
      for (uint i = 0; i < local_size; ++i)
      {
        outfile << indent()
          << "<element index=\"" << indices[i]
          << "\" value=\"" << std::setprecision(16) << vector_values[i]
          << "\"/>" << std::endl;
      }
    }
  }

  if (write_to_stream)
  {
    --indent;
    outfile << indent() << "</array>" << std::endl;
    --indent;
  }

  // Write vector footer
  if (write_to_stream)
    outfile << indent() << "</vector>" << std::endl;
  */
}
//-----------------------------------------------------------------------------
void XMLVector::read_vector_tag(const xmlChar *name, const xmlChar **attrs)
{
  state = INSIDE_VECTOR;
}
//-----------------------------------------------------------------------------
void XMLVector::read_array_tag(const xmlChar *name, const xmlChar **attrs)
{
  xml_array.reset(new XMLArray(values, parser, true));
  xml_array->read_array_tag(name, attrs);
  xml_array->handle();
}
//-----------------------------------------------------------------------------
void XMLVector::end_vector()
{
  assert(xml_array);

  // Resize vector
  x.resize(xml_array->size);

  // Set values in the vector
  x.set(&values[0], xml_array->element_index.size(),
        &(xml_array->element_index)[0]);
  x.apply("insert");
}
//-----------------------------------------------------------------------------
