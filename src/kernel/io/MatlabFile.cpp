// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Erik Svensson, 2003.
// Modified by Andy R. Terrel, 2005.
//
// First added:  2003-02-17
// Last changed: 2005-12-30

#include <stdio.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/Matrix.h>
#include <dolfin/MatlabFile.h>

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
  // Get PETSc Mat pointer
  Mat A_mat = A.mat();

  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Write matrix in sparse format
  fprintf(fp, "%s = [", A.name().c_str());
  int ncols = 0;
  const int *cols = 0;
  const double *vals = 0;
  for (uint i = 0; i < A.size(0); i++)
  {
    MatGetRow(A_mat, i, &ncols, &cols, &vals);
    for (int pos = 0; pos < ncols; pos++)
    {
      fprintf(fp, " %i %i %.16e", i + 1, cols[pos] + 1, vals[pos]);		
      if ( i == (A.size(0) - 1) && (pos + 1 == ncols) )
	fprintf(fp, "];\n");
      else {
	fprintf(fp, "\n");
      }
    }
    MatRestoreRow(A_mat, i, &ncols, &cols, &vals);
  }
  fprintf(fp, "%s = spconvert(%s);\n", A.name().c_str(), A.name().c_str());
  
  // Close file
  fclose(fp);

  cout << "Saved matrix " << A.name() << " (" << A.label()
       << ") to file " << filename << " in sparse MATLAB format." << endl;
}
//-----------------------------------------------------------------------------
