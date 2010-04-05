// Copyright (C) 2003-2008 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Erik Svensson, 2003.
// Modified by Andy R. Terrel, 2005.
// Modified by Garth N. Wells, 2006.
//
// First added:  2003-02-17
// Last changed: 2010-04-05

#include <stdio.h>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/la/GenericMatrix.h>
#include "MatlabFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MatlabFile::MatlabFile(const std::string filename) : MFile(filename)
{
  type = "MATLAB";
}
//-----------------------------------------------------------------------------
MatlabFile::~MatlabFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MatlabFile::operator<<(const GenericMatrix& A)
{
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  if (!fp)
    error("Unable to open file %s", filename.c_str());

  // Write matrix in sparse format
  fprintf(fp, "A = [");

  std::vector<uint> columns;
  std::vector<double> values;

  for (uint i = 0; i < A.size(0); i++)
  {
    A.getrow(i, columns, values);
    for (uint pos = 0; pos < columns.size(); pos++)
    {
      fprintf(fp, " %u %i %.15g", i + 1, columns[pos] + 1, values[pos]);
      if ( i == (A.size(0) - 1) && (pos + 1 == columns.size()) )
        fprintf(fp, "];\n");
      else
        fprintf(fp, "\n");
    }
  }
  fprintf(fp, "A = spconvert(A);\n");

  // Close file
  fclose(fp);

  info(TRACE, "Saved matrix to file %s in sparse MATLAB format", filename.c_str());
}
//-----------------------------------------------------------------------------

