// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-07-15
// Last changed: 2008-06-14

#ifndef __XML_MATRIX_H
#define __XML_MATRIX_H

#include <dolfin/la/GenericMatrix.h>
#include "XMLObject.h"

namespace dolfin
{

  class XMLMatrix : public XMLObject
  {
  public:

    XMLMatrix(GenericMatrix& matrix);

    void start_element (const xmlChar *name, const xmlChar **attrs);
    void end_element   (const xmlChar *name);

  private:

    enum ParserState { OUTSIDE, INSIDE_MATRIX, INSIDE_ROW, DONE };

    void read_matrix  (const xmlChar *name, const xmlChar **attrs);
    void read_row     (const xmlChar *name, const xmlChar **attrs);
    void read_entry   (const xmlChar *name, const xmlChar **attrs);

    void set_row();

    GenericMatrix& A;
    ParserState state;

    // Data for row
    uint row;
    std::vector<uint> columns;
    std::vector<double> values;

  };

}

#endif
