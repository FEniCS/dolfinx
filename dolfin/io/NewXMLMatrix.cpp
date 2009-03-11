// Copyright (C) 2009 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-11
// Last changed: 2009-03-11

#include <dolfin/la/GenericMatrix.h>
#include <dolfin/log/dolfin_log.h>
#include "NewXMLMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NewXMLMatrix::NewXMLMatrix(GenericMatrix& matrix, NewXMLFile& parser)
  : XMLHandler(parser), A(matrix), state(OUTSIDE), row(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewXMLMatrix::start_element(const xmlChar *name, const xmlChar **attrs)
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
void NewXMLMatrix::end_element(const xmlChar *name)
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
void NewXMLMatrix::write(const GenericMatrix& A, std::ostream& outfile, uint indentation_level)
{
  uint curr_indent = indentation_level;


  // Write matrix header
  outfile << std::setw(curr_indent) << "";
  outfile << "<matrix rows=\"" << A.size(0) << "\" columns=\"" << A.size(1) << "\">" << std::endl;

  // Write matrix content
  std::vector<uint> columns;
  std::vector<double> values;
  for (unsigned int i = 0; i < A.size(0); i++)
  {
    A.getrow(i, columns, values);
    if (columns.size() > 0)
    {
      curr_indent = indentation_level + 2;
      outfile << std::setw(curr_indent) << "";
      // Write row header
      outfile << "<row index=\"" << i << "\" size=\"" << columns.size() << "\">" << std::endl;

      curr_indent = indentation_level + 4;
      for (uint pos = 0; pos < columns.size(); pos++)
      {
        unsigned int j = columns[pos];
        double aij = values[pos];
        outfile << std::setw(curr_indent) << "";
        outfile << "<entry column=\"" << j <<"\" value=\"" << aij << "\"/>" << std::endl;
      }
      // Write row footer
      curr_indent = indentation_level + 2;
      outfile << std::setw(curr_indent) << "";
      outfile << "</row>" << std::endl;
    }
  }
  // Write matrix footer
  curr_indent = indentation_level;
  outfile << "<matrix rows=\"" << A.size(0) << "\" columns=\"" << A.size(1) << "\">" << std::endl;
  outfile << "</matrix>" << std::endl;
}
//-----------------------------------------------------------------------------
void NewXMLMatrix::read_matrix(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint M = parse_int(name, attrs, "rows");
  uint N = parse_int(name, attrs, "columns");

  // Set size
  A.resize(M, N);
}
//-----------------------------------------------------------------------------
void NewXMLMatrix::read_row(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  row = parse_uint(name, attrs, "index");
  const uint size = parse_uint(name, attrs, "size");

  // Reset data for row
  columns.clear();
  values.clear();
  columns.reserve(size);
  values.reserve(size);
}
//-----------------------------------------------------------------------------
void NewXMLMatrix::read_entry(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  const uint column = parse_uint(name, attrs, "column");
  const double value  = parse_float(name, attrs, "value");
  
  // Set values
  columns.push_back(column);
  values.push_back(value);
}
//-----------------------------------------------------------------------------
void NewXMLMatrix::set_row()
{
  A.setrow(row, columns, values);
}
//-----------------------------------------------------------------------------
