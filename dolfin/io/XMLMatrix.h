// Copyright (C) 2009 Ola Skavhaug and Anders Logg
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
// First added:  2009-03-11
// Last changed: 2009-03-11

#ifndef __XMLMATRIX_H
#define __XMLMATRIX_H

#include "XMLHandler.h"

namespace dolfin
{
  class GenericMatrix;
  class XMLFile;

  class XMLMatrix : public XMLHandler
  {
  public:

    XMLMatrix(GenericMatrix& matrix, XMLFile& parser);

    void start_element (const xmlChar *name, const xmlChar **attrs);
    void end_element   (const xmlChar *name);

    static void write(const GenericMatrix& matrix, std::ostream& outfile, uint indentation_level=0);

  private:

    enum parser_state { OUTSIDE, INSIDE_MATRIX, INSIDE_ROW, DONE };

    void read_matrix  (const xmlChar *name, const xmlChar **attrs);
    void read_row     (const xmlChar *name, const xmlChar **attrs);
    void read_entry   (const xmlChar *name, const xmlChar **attrs);

    void set_row();

    GenericMatrix& A;
    parser_state state;

    // Data for row
    uint row;
    std::vector<uint> columns;
    std::vector<double> values;

  };

}

#endif
