// Copyright (C) 2003-2007 Anders Logg and Ola Skavhaug
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

#ifndef __XML_VECTOR_H
#define __XML_VECTOR_H

#include <boost/scoped_ptr.hpp>
#include "XMLArray.h"
#include "XMLHandler.h"

namespace dolfin
{

  /// Forward declarations
  class GenericVector;

  class XMLVector : public XMLHandler
  {
  public:

    /// Constructor
    XMLVector(GenericVector& vector, XMLFile& parser);

    /// Destructor
    ~XMLVector();

    void start_element(const xmlChar *name, const xmlChar **attrs);
    void end_element(const xmlChar *name);

    /// Write the XML file
    static void write(const GenericVector& vector, std::ostream& outfile,
                      uint indentation_level=0);

    /// Read the vector begin tag
    void read_vector_tag(const xmlChar *name, const xmlChar **attrs);

  private:

    enum parser_state { OUTSIDE, INSIDE_VECTOR, DONE };

    void end_vector();
    void read_array_tag(const xmlChar *name, const xmlChar **attrs);

    GenericVector& x;
    parser_state state;

    std::vector<double> values;
    boost::scoped_ptr<XMLArray> xml_array;

  };

}

#endif
