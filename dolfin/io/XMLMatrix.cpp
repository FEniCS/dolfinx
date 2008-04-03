// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-02-17
// Last changed: 2006-05-23


#include <dolfin/la/GenericMatrix.h>
#include <dolfin/log/dolfin_log.h>
#include "XMLMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLMatrix::XMLMatrix(GenericMatrix& matrix) : XMLObject(), A(matrix)
{
  state = OUTSIDE;
  row = 0;
}
//-----------------------------------------------------------------------------
void XMLMatrix::startElement(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "sparsematrix") == 0 )
    {
      readMatrix(name, attrs);
      state = INSIDE_MATRIX;
    }
    
    break;
    
  case INSIDE_MATRIX:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "row") == 0 )
    {
      readRow(name, attrs);
      state = INSIDE_ROW;
    }
    
    break;
    
  case INSIDE_ROW:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "element") == 0 )
      readEntry(name, attrs);
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMatrix::endElement(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_MATRIX:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "sparsematrix") == 0 )
      state = DONE;
    
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
  // Parse values
  uint M = parseInt(name, attrs, "rows");
  uint N = parseInt(name, attrs, "columns");

  // Set values
  A.init(M, N);
}
//-----------------------------------------------------------------------------
void XMLMatrix::readRow(const xmlChar *name, const xmlChar **attrs)
{
  // FIXME: update to new format
  error("This function needs to be updated to the new format.");

  /*
  // Set default values
  row = 0;
  int size = 0;
  
  // Parse values
  parseIntegerRequired(name, attrs, "row",  row);
  parseIntegerRequired(name, attrs, "size", size);

  // Set values
  A.initrow(row, size);
  */
}
//-----------------------------------------------------------------------------
void XMLMatrix::readEntry(const xmlChar *name, const xmlChar **attrs)
{
  error("XMLMatrix::readEntry needs to updated for new matrix element access.");
/*
  // Parse values
  uint column = parseUnsignedInt(name, attrs, "column");
  real value  = parseReal(name, attrs, "value");
  
  // FIXME: update to matrix element access
  // Set values
  A(row, column) = value;
*/
}
//-----------------------------------------------------------------------------
