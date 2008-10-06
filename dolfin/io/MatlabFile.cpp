// Copyright (C) 2003-2008 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Erik Svensson, 2003.
// Modified by Andy R. Terrel, 2005.
// Modified by Garth N. Wells, 2006.
//
// First added:  2003-02-17
// Last changed: 2008-04-23

#include <stdio.h>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/la/GenericMatrix.h>
#include "MatlabFile.h"
#include <dolfin/common/Array.h>

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
void MatlabFile::operator<<(GenericMatrix& A)
{
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Write matrix in sparse format
//  fprintf(fp, "%s = [", A.name().c_str());
  fprintf(fp, "A = [");

  Array<uint> columns;
  Array<double> values;
  
  for (uint i = 0; i < A.size(0); i++)
  {
    A.getrow(i, columns, values);
    for (uint pos = 0; pos < columns.size(); pos++)
    {
      fprintf(fp, " %u %i %.15g", i + 1, columns[pos] + 1, values[pos]);
      if ( i == (A.size(0) - 1) && (pos + 1 == columns.size()) )
        fprintf(fp, "];\n");
      else {
        fprintf(fp, "\n");
      }
    }
  }
//  fprintf(fp, "%s = spconvert(%s);\n", A.name().c_str(), A.name().c_str());
  fprintf(fp, "A = spconvert(A);\n");
  
  // Close file
  fclose(fp);

//  message(1, "Saved matrix %s (%s) to file %s in sparse MATLAB format",
//          A.name().c_str(), A.label().c_str(), filename.c_str());
  message(1, "Saved matrix to file %s in sparse MATLAB format", filename.c_str());
}
//-----------------------------------------------------------------------------

