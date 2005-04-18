// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// New output format for Matrix added by Erik Svensson 2003

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
  dolfin_error("This function needs to be updated to the new format.");

  /*
  // Octave file format for Matrix is not the same as the Matlab format,
  // since octave cannot handle sparse matrices.
  
  real value;
  
  FILE *fp = fopen(filename.c_str(), "a");
  
  fprintf(fp, "%s = [", A.name().c_str());
  for (unsigned int i = 0; i < A.size(0); i++) {

    for (unsigned int j = 0; j < A.size(1); j++) {
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

  // Increase the number of times we have saved the matrix
  ++A;

  cout << "Saved matrix " << A.name() << " (" << A.label()
       << ") to file " << filename << " in Octave format." << endl;
  */
}
//-----------------------------------------------------------------------------
