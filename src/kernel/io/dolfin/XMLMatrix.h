// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __XML_MATRIX_H
#define __XML_MATRIX_H

#include "XMLObject.h"

namespace dolfin {

  class Matrix;
  
  class XMLMatrix : public XMLObject {
  public:

	 XMLMatrix(Matrix& matrix);
	 
	 void startElement (const xmlChar *name, const xmlChar **attrs);
	 void endElement   (const xmlChar *name);
	 
  private:

	 enum ParserState { OUTSIDE, INSIDE_MATRIX, INSIDE_ROW, DONE };
	 
	 void readMatrix  (const xmlChar *name, const xmlChar **attrs);
	 void readRow     (const xmlChar *name, const xmlChar **attrs);
	 void readElement (const xmlChar *name, const xmlChar **attrs);
	 
	 Matrix& A;
	 int row;
	 
	 ParserState state;
	 
  };
  
}

#endif
