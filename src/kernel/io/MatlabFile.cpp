// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// New output format for Matrix added by Erik Svensson 2003

// FIXME: Use streams rather than stdio
#include <stdio.h>

#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Grid.h>

#include "MatlabFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MatlabFile::operator>>(Vector& x)
{
  // FIXME: Use logging system
  cout << "Warning: Cannot read vectors from Matlab files." << endl;
}
//-----------------------------------------------------------------------------
void MatlabFile::operator>>(Matrix& A)
{
  // FIXME: Use logging system
  cout << "Warning: Cannot read matrices from Matlab files." << endl;
}
//-----------------------------------------------------------------------------
void MatlabFile::operator>>(Grid& grid)
{
  // FIXME: Use logging system
  cout << "Warning: Cannot read grids from Matlab files." << endl;
}
//-----------------------------------------------------------------------------
void MatlabFile::operator<<(const Vector& x)
{
  FILE *fp = fopen(filename.c_str(), "a");

  fprintf(fp, "x = [");
  for (int i = 0; i < x.size(); i++)
	 fprintf(fp, " %.16e", x(i));
  fprintf(fp, " ];\n");

  fclose(fp);
}
//-----------------------------------------------------------------------------
void MatlabFile::operator<<(const Matrix& A)
{
  real value;
  int j ;
  
  FILE *fp = fopen(filename.c_str(), "a");
  
  fprintf(fp, "A = [");
  for (int i = 0; i < A.size(0); i++) {
    for (int pos = 0; !A.endrow(i, pos); pos++) {
      value = A(i, &j, pos);
		fprintf(fp, " %i %i %.16e", i + 1, j + 1, value);		
		if ( i == (A.size(0) - 1) && A.endrow(i, pos + 1) )
        fprintf(fp, "];\n");
		else {
		  fprintf(fp, "\n");
		}
    }
  }
  fprintf(fp, "A=spconvert(A);\n");
  
  fclose(fp);
}
//-----------------------------------------------------------------------------
void MatlabFile::operator<<(const Grid& Grid)
{
  cout << "Warning: Cannot save grids to Matlab files." << endl;
}
//-----------------------------------------------------------------------------
