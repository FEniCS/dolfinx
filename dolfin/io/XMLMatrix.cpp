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
void XMLMatrix::start_element(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:

    if ( xmlStrcasecmp(name, (xmlChar *) "matrix") == 0 )
    {
      read_matrix(name, attrs);
      state = INSIDE_MATRIX;
    }

    break;

  case INSIDE_MATRIX:

    if ( xmlStrcasecmp(name, (xmlChar *) "row") == 0 )
    {
      read_row(name, attrs);
      state = INSIDE_ROW;
    }

    break;

  case INSIDE_ROW:

    if ( xmlStrcasecmp(name, (xmlChar *) "entry") == 0 )
      read_entry(name, attrs);

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMatrix::end_element(const xmlChar *name)
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
      set_row();
      state = INSIDE_MATRIX;
    }

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMatrix::read_matrix(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint M = parse_int(name, attrs, "rows");
  uint N = parse_int(name, attrs, "columns");

  // Set size
  A.resize(M, N);
}
//-----------------------------------------------------------------------------
void XMLMatrix::read_row(const xmlChar *name, const xmlChar **attrs)
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
void XMLMatrix::read_entry(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  const uint column = parseUnsignedInt(name, attrs, "column");
  const double value  = parse_real(name, attrs, "value");

  // Set values
  columns.push_back(column);
  values.push_back(value);
}
//-----------------------------------------------------------------------------
void XMLMatrix::set_row()
{
  A.setrow(row, columns, values);
}
//-----------------------------------------------------------------------------
