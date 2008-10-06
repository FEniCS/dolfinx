// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-07-15
// Last changed: 2008-06-14

#ifndef __XML_MATRIX_H
#define __XML_MATRIX_H

#include <dolfin/common/Array.h>
#include <dolfin/la/GenericMatrix.h>
#include "XMLObject.h"

namespace dolfin
{
  
  class XMLMatrix : public XMLObject
  {
  public:

    XMLMatrix(GenericMatrix& matrix);
    
    void startElement (const xmlChar *name, const xmlChar **attrs);
    void endElement   (const xmlChar *name);
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_MATRIX, INSIDE_ROW, DONE };
    
    void readMatrix  (const xmlChar *name, const xmlChar **attrs);
    void readRow     (const xmlChar *name, const xmlChar **attrs);
    void readEntry   (const xmlChar *name, const xmlChar **attrs);

    void setRow();
    
    GenericMatrix& A;
    ParserState state;

    // Data for row
    uint row;
    Array<uint> columns;
    Array<double> values;
    
  };
  
}

#endif
