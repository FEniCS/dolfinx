// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Erik Svensson, 2003.
//
// First added:  2003-02-26
// Last changed: 2005-11-15

// FIXME: Use streams rather than stdio
#include <stdio.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/MatlabFile.h>
#include <dolfin/OctaveFile.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
OctaveFile::OctaveFile(const std::string filename) : MFile(filename)
{
  type = "Octave";
}
//-----------------------------------------------------------------------------
OctaveFile::~OctaveFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void OctaveFile::operator<<(Matrix& A)
{
  // Octave file format for Matrix is not the same as the Matlab format,
  // since octave cannot handle sparse matrices.

  uint M = A.size(0);
  uint N = A.size(1);
  real* row = new real[N];
  
  FILE *fp = fopen(filename.c_str(), "a");
  fprintf(fp, "%s = [", A.name().c_str());

  for (uint i = 0; i < M; i++)
  {
    // Reset entries on row
    for (uint j = 0; j < N; j++)
      row[j] = 0.0;

    // Get nonzero entries
    int ncols = 0;
    const int* cols = 0;
    const double* vals = 0;
    MatGetRow(A.mat(), i, &ncols, &cols, &vals);
    for (int pos = 0; pos < ncols; pos++)
      row[cols[pos]] = vals[pos];
    MatRestoreRow(A.mat(), i, &ncols, &cols, &vals);

    // Write row
    for (uint j = 0; j < N; j++)
    {
      if ( row[j] == 0.0 )
        fprintf(fp, " 0");
      else
        fprintf(fp, " %.15e", row[j]);
    }

    // New line or end of matrix
    if ( i == (M - 1) )
      fprintf(fp, "];\n");
    else
      fprintf(fp, "\n");
  }
  
  fclose(fp);
  delete [] row;
  
  cout << "Saved matrix " << A.name() << " (" << A.label()
       << ") to file " << filename << " in Octave format." << endl;
}
//-----------------------------------------------------------------------------
