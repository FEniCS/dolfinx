// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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
  FILE *fp = fopen(filename.c_str(), "a");

  fprintf(fp, "A = [");
  for (int i = 0; i < A.size(0); i++) {
	 for (int j = 0; j < A.size(1); j++)
		fprintf(fp, " %.16e", A(i,j));
	 if ( i == (A.size(0) - 1) )
		fprintf(fp,"];\n");
	 else
		fprintf(fp,"\n");
  }

  fclose(fp);
}
//-----------------------------------------------------------------------------
void MatlabFile::operator<<(const Grid& Grid)
{
  cout << "Warning: Cannot save grids to Matlab files." << endl;
}
//-----------------------------------------------------------------------------
