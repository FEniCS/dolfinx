// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Erik Svensson, 2003.
//
// First added:  2003-02-17
// Last changed: 2005

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
  // FIXME: update to new format
  dolfin_error("This function needs to be updated to the new format.");

  /*
  real value;
  unsigned int j;

  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  case Matrix::sparse:
  
  Usage:

  File file("matrix.m");
  file << A;
  
  Fix this Andy... :-)
  
  Call A.mat()
  Use MatGetRow(A, i, &ncols, &cols, &vals);
  Look in Matrix::disp()

    // Write matrix in sparse format
    fprintf(fp, "%s = [", A.name().c_str());
    for (unsigned int i = 0; i < A.size(0); i++) {
      for (unsigned int pos = 0; !A.endrow(i, pos); pos++) {
	value = A(i, j, pos);
	fprintf(fp, " %i %i %.16e", i + 1, j + 1, value);		
	if ( i == (A.size(0) - 1) && A.endrow(i, pos + 1) )
	  fprintf(fp, "];\n");
	else {
	  fprintf(fp, "\n");
	}
      }
    }
    fprintf(fp, "%s = spconvert(%s);\n", A.name().c_str(), A.name().c_str());
    
    break;

  // Close file
  fclose(fp);

  cout << "Saved matrix " << A.name() << " (" << A.label()
       << ") to file " << filename << " in Matlab format." << endl;
  */
}
//-----------------------------------------------------------------------------
