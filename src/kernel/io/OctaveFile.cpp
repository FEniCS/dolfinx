// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// New output format for Matrix added by Erik Svensson 2003

// FIXME: Use streams rather than stdio
#include <stdio.h>

#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Grid.h>
#include <dolfin/Function.h>

#include "MatlabFile.h"
#include "OctaveFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void OctaveFile::operator>>(Vector& x)
{
  // FIXME: Use logging system
  std::cout << "Warning: Cannot read vectors from Octave files." << std::endl;
}
//-----------------------------------------------------------------------------
void OctaveFile::operator>>(Matrix& A)
{
  // FIXME: Use logging system
  std::cout << "Warning: Cannot read matrices from Octave files." << std::endl;
}
//-----------------------------------------------------------------------------
void OctaveFile::operator>>(Grid& grid)
{
  // FIXME: Use logging system
  std::cout << "Warning: Cannot read grids from Octave files." << std::endl;
}
//-----------------------------------------------------------------------------
void OctaveFile::operator>>(Function& u)
{
  // FIXME: Use logging system
  std::cout << "Warning: Cannot read functions from Octave files." << std::endl;
}
//-----------------------------------------------------------------------------
void OctaveFile::operator<<(const Vector& x)
{
  MatlabFile::writeVector(x, filename, x.name(), x.label());
  std::cout << "Saved vector " << x.name() << " (" << x.label()
				<< ") to file " << filename << " in Octave format." << std::endl;
}
//-----------------------------------------------------------------------------
void OctaveFile::operator<<(const Matrix& A)
{
  // Octave file format for Matrix is not the same as the Matlab format,
  // since octave cannot handle sparse matrices.
  
  real value;
  
  FILE *fp = fopen(filename.c_str(), "a");
  
  fprintf(fp, "%s = [", A.name().c_str());
  for (int i = 0; i < A.size(0); i++) {

    for (int j = 0; j < A.size(1); j++) {
		if ( (value = A(i,j)) == 0.0 )
		  fprintf(fp, " 0");
		else
		  fprintf(fp, " %.16e", value);
	 }
	 
	 if ( i == (A.size(0) - 1) )
		fprintf(fp, "];\n");
	 else
		fprintf(fp, "\n");
	 
  }
  
  fclose(fp);

  std::cout << "Saved matrix " << A.name() << " (" << A.label()
				<< ") to file " << filename << " in Octave format." << std::endl;
}
//-----------------------------------------------------------------------------
void OctaveFile::operator<<(const Grid& grid)
{
  MatlabFile::writeGrid(grid, filename, grid.name(), grid.label());
  std::cout << "Saved grid " << grid.name() << " (" << grid.label()
				<< ") to file " << filename << " in Octave format." << std::endl;
}
//-----------------------------------------------------------------------------
void OctaveFile::operator<<(const Function& u)
{
  MatlabFile::writeFunction(u, filename, u.name(), u.label());
  std::cout << "Saved function " << u.name() << " (" << u.label()
				<< ") to file " << filename << " in Octave format." << std::endl;
}
//-----------------------------------------------------------------------------
