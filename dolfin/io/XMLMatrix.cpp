// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-02-17
// Last changed: 2008-06-15

#include <dolfin/la/GenericMatrix.h>
#include <dolfin/log/dolfin_log.h>
#include "XMLMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLMatrix::XMLMatrix(GenericMatrix& matrix)
  : XMLObject(), A(matrix), state(OUTSIDE), row(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLMatrix::startElement(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "matrix") == 0 )
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
    
    if ( xmlStrcasecmp(name, (xmlChar *) "entry") == 0 )
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
    
    if ( xmlStrcasecmp(name,(xmlChar *) "matrix") == 0 )
    {
      A.apply();
      state = DONE;
    }
    
    break;
    
  case INSIDE_ROW:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "row") == 0 )
    {
      setRow();
      state = INSIDE_MATRIX;
    }
    
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

  // Set size
  A.resize(M, N);
}
//-----------------------------------------------------------------------------
void XMLMatrix::readRow(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  row = parseUnsignedInt(name, attrs, "row");
  const uint size = parseUnsignedInt(name, attrs, "size");

  // Reset data for row
  columns.clear();
  values.clear();
  columns.reserve(size);
  values.reserve(size);
}
//-----------------------------------------------------------------------------
void XMLMatrix::readEntry(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  const uint column = parseUnsignedInt(name, attrs, "column");
  const real value  = parseReal(name, attrs, "value");
  
  // Set values
  columns.push_back(column);
  values.push_back(value);
}
//-----------------------------------------------------------------------------
void XMLMatrix::setRow()
{
  A.setrow(row, columns, values);
}
//-----------------------------------------------------------------------------
