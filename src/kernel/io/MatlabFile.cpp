// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// New output format for Matrix added by Erik Svensson 2003

#include <stdio.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/Matrix.h>
#include <dolfin/TimeSlabSample.h>
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
  real value;
  unsigned int j ;

  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  switch ( A.type() ) {
  case Matrix::dense:

    for (unsigned int i = 0; i < A.size(0); i++) {
      for (unsigned int j = 0; j<A.size(1); j++) {
        fprintf(fp, "%.16e ", A[i][j]);
        if ( j < (A.size(1) - 1) && i <= (A.size(0) - 1))
          fprintf(fp, ", ");
        else if ( j == (A.size(1) - 1) && i < (A.size(0) - 1))
          fprintf(fp, ";\n");
        else
          fprintf(fp, "];\n");
      }
    }

    break;

  case Matrix::sparse:

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
  
  default:
    dolfin_error("Unknown matrix format.");
  }

  // Close file
  fclose(fp);

  // Increase the number of times we have saved the matrix
  ++A;
  
  cout << "Saved matrix " << A.name() << " (" << A.label()
       << ") to file " << filename << " in Matlab format." << endl;
}
//-----------------------------------------------------------------------------
void MatlabFile::operator<<(TimeSlabSample& sample)
{
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Initialize data structures first time
  if ( no_frames == 0 )
  {
    fprintf(fp, "t = [];\n");
    fprintf(fp, "u = [];\n");
    fprintf(fp, "k = [];\n");
    fprintf(fp, "r = [];\n");
    fprintf(fp, "\n");
  }

  // Save time
  fprintf(fp, "t = [t %.16e];\n", sample.time());

  // Save solution
  fprintf(fp, "tmp = [ ");
  for (unsigned int i = 0; i < sample.size(); i++)
    fprintf(fp, "%.16e ", sample.value(i));  
  fprintf(fp, "];\n");
  fprintf(fp, "u = [u tmp'];\n");
  fprintf(fp, "clear tmp;\n");

  // Save time steps
  fprintf(fp, "tmp = [ ");
  for (unsigned int i = 0; i < sample.size(); i++)
    fprintf(fp, "%.16e ", sample.timestep(i));
  fprintf(fp, "];\n");
  fprintf(fp, "k = [k tmp'];\n");
  fprintf(fp, "clear tmp;\n");

  // Save residuals
  fprintf(fp, "tmp = [ ");
  for (unsigned int i = 0; i < sample.size(); i++)
    fprintf(fp, "%.16e ", sample.residual(i));
  fprintf(fp, "];\n");
  fprintf(fp, "r = [r tmp'];\n");
  fprintf(fp, "clear tmp;\n");

  // Increase frame counter
  no_frames++;
  
  // Close file
  fclose(fp);
}
//-----------------------------------------------------------------------------
