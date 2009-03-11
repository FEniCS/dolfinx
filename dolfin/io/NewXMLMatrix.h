// Copyright (C) 2009 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-11
// Last changed: 2009-03-11

#ifndef __NEWNewXMLMATRIX_H
#define __NEWNewXMLMATRIX_H

#include <dolfin/la/GenericMatrix.h>
#include "XMLHandler.h"

namespace dolfin
{
  
  class NewXMLMatrix : public XMLHandler
  {
  public:

    NewXMLMatrix(GenericMatrix& matrix, NewXMLFile& parser);
    
    void start_element (const xmlChar *name, const xmlChar **attrs);
    void end_element   (const xmlChar *name);

    static void write(const GenericMatrix& matrix, std::ofstream& outfile, uint indentation_level=0);
    
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
