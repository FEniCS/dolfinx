// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __XML_SPARSE_MATRIX_H
#define __XML_SPARSE_MATRIX_H

#include "XMLObject.h"

namespace dolfin {

  class SparseMatrix;
  
  class XMLSparseMatrix : public XMLObject {
  public:

	 XMLSparseMatrix(SparseMatrix *sparseMatrix);
	 
	 void startElement (const xmlChar *name, const xmlChar **attrs);
	 void endElement   (const xmlChar *name);
	 
  private:

	 enum ParserState { OUTSIDE, INSIDE_SPARSE_MATRIX, INSIDE_ROW, DONE };
	 
	 void readSparseMatrix (const xmlChar *name, const xmlChar **attrs);
	 void readRow          (const xmlChar *name, const xmlChar **attrs);
	 void readElement      (const xmlChar *name, const xmlChar **attrs);
	 
	 SparseMatrix *sparseMatrix;
	 int row;
	 
	 ParserState state;
	 
  };
  
}

#endif
