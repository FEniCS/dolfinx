// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-05-06
// Last changed: 2005-09-20

#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Mesh.h>
#include <dolfin/Function.h>
#include <dolfin/Sample.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/PythonFile.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PythonFile::PythonFile(const std::string filename) : 
  GenericFile(filename)
{
  type = "Python";

  filename_t = filename + ".t";
  filename_u = filename + ".u";
  filename_k = filename + ".k";
  filename_r = filename + ".r";
}
//-----------------------------------------------------------------------------
PythonFile::~PythonFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PythonFile::operator<<(Sample& sample)
{
  FILE* fp_t = 0;
  FILE* fp_u = 0;
  FILE* fp_k = 0;
  FILE* fp_r = 0;

  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Open sub-files
  if ( no_frames == 0 )
  {
    // Open files (first time)
    fp_t = fopen(filename_t.c_str(), "w");
    fp_u = fopen(filename_u.c_str(), "w");
    fp_k = fopen(filename_k.c_str(), "w");
    fp_r = fopen(filename_r.c_str(), "w");
  }
  else
  {
    // Open files
    fp_t = fopen(filename_t.c_str(), "a");
    fp_u = fopen(filename_u.c_str(), "a");
    fp_k = fopen(filename_k.c_str(), "a");
    fp_r = fopen(filename_r.c_str(), "a");
  }

  // Python wrapper file
  if ( no_frames == 0 )
  {
    fprintf(fp, "from Numeric import *\n");
    fprintf(fp, "from scipy import *\n");
    fprintf(fp, "import dolfin\n");
    fprintf(fp, "\n");
    fprintf(fp, "fp_t = open(\"%s\")\n", filename_t.c_str());
    fprintf(fp, "fp_u = open(\"%s\")\n", filename_u.c_str());
    fprintf(fp, "fp_k = open(\"%s\")\n", filename_k.c_str());
    fprintf(fp, "fp_r = open(\"%s\")\n", filename_r.c_str());
    fprintf(fp, "t = dolfin.read_array(fp_t)\n");
    fprintf(fp, "u = dolfin.read_array(fp_u)\n");
    fprintf(fp, "k = dolfin.read_array(fp_k)\n");
    fprintf(fp, "r = dolfin.read_array(fp_r)\n");
  }

  // Save time
  fprintf(fp_t, "%.15e ", sample.t());

  // Save solution
  for (unsigned int i = 0; i < sample.size(); i++)
  {
    fprintf(fp_u, "%.15e ", sample.u(i));  
  }
  fprintf(fp_u, "\n");

  // Save time steps
  for (unsigned int i = 0; i < sample.size(); i++)
  {
    fprintf(fp_k, "%.15e ", sample.k(i));  
  }
  fprintf(fp_k, "\n");

  // Save residuals
  for (unsigned int i = 0; i < sample.size(); i++)
  {
    fprintf(fp_r, "%.15e ", sample.r(i));  
  }
  fprintf(fp_r, "\n");


  // Increase frame counter
  no_frames++;
  
  // Close files
  fclose(fp_t);
  fclose(fp_u);
  fclose(fp_k);
  fclose(fp_r);

  // Close file
  fclose(fp);

}
//-----------------------------------------------------------------------------
