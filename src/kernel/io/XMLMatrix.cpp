// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Matrix.h>
#include "XMLMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLMatrix::XMLMatrix(Matrix& matrix) : XMLObject(), A(matrix)
{
  state = OUTSIDE;
  row = 0;
}
//-----------------------------------------------------------------------------
void XMLMatrix::startElement(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state ){
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "sparsematrix") == 0 ) {
      readMatrix(name,attrs);
      state = INSIDE_MATRIX;
    }
    
    break;
    
  case INSIDE_MATRIX:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "row") == 0 ) {
      readRow(name,attrs);
      state = INSIDE_ROW;
    }
    
    break;
    
  case INSIDE_ROW:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "element") == 0 )
      readElement(name,attrs);
    
    break;
    
  default:
    ;
  }
  
}
//-----------------------------------------------------------------------------
void XMLMatrix::endElement(const xmlChar *name)
{
  switch ( state ){
  case INSIDE_MATRIX:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "sparsematrix") == 0 ) {
      ok = true;
      state = DONE;
    }
    
    break;
    
  case INSIDE_ROW:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "row") == 0 )
      state = INSIDE_MATRIX;
    
    break;
    
  default:
    ;
  }
  
  
}
//-----------------------------------------------------------------------------
void XMLMatrix::readMatrix(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int rows = 0;
  int columns = 0;

  // Parse values
  parseIntegerRequired(name, attrs, "rows",    &rows);
  parseIntegerRequired(name, attrs, "columns", &columns);

  // Set values
  A.init(rows, columns);
}
//-----------------------------------------------------------------------------
void XMLMatrix::readRow(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  row = 0;
  int size = 0;
  
  // Parse values
  parseIntegerRequired(name, attrs, "row", &row);
  parseIntegerRequired(name, attrs, "size", &size);

  // Set values
  A.initrow(row, size);
}
//-----------------------------------------------------------------------------
void XMLMatrix::readElement(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int column = 0;
  real value = 0.0;
  
  // Parse values
  parseIntegerRequired (name, attrs, "column", &column);
  parseRealRequired    (name, attrs, "value",  &value);
  
  // Set values
  A(row,column) = value;
}
//-----------------------------------------------------------------------------
