// Copyright (C) 2003-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Erik Svensson, 2003.
// Modified by Andy R. Terrel, 2005.
// Modified by Garth N. Wells, 2006.
//
// First added:  2003-02-17
// Last changed: 2006-05-30

#include <stdio.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/Matrix.h>
#include <dolfin/MatlabFile.h>
#include <dolfin/Array.h>

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
void MatlabFile::operator<<(Matrix& A)
{
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Write matrix in sparse format
  fprintf(fp, "%s = [", A.name().c_str());

  int ncols = 0;
  Array<int> columns;
  Array<real> values;
  
  for (uint i = 0; i < A.size(0); i++)
  {
    A.getRow(i, ncols, columns, values);
    for (int pos = 0; pos < ncols; pos++)
    {
      fprintf(fp, " %u %i %.15g", i + 1, columns[pos] + 1, values[pos]);
      if ( i == (A.size(0) - 1) && (pos + 1 == ncols) )
        fprintf(fp, "];\n");
      else {
        fprintf(fp, "\n");
      }
    }
  }
  fprintf(fp, "%s = spconvert(%s);\n", A.name().c_str(), A.name().c_str());
  
  // Close file
  fclose(fp);


  cout << "Saved matrix " << A.name() << " (" << A.label()
       << ") to file " << filename << " in sparse MATLAB format." << endl;
}
//-----------------------------------------------------------------------------

