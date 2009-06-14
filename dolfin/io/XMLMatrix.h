// Copyright (C) 2009 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
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
