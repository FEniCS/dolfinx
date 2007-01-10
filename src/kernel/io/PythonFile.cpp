// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2003-05-06
// Last changed: 2006-05-29

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
PythonFile::PythonFile(const std::string filename) : GenericFile(filename)
{
  type = "Python";

  std::string prefix = filename.substr(0, filename.rfind("."));
  filename_t = prefix + "_t.data";
  filename_u = prefix + "_u.data";
  filename_k = prefix + "_k.data";
  filename_r = prefix + "_r.data";
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
  if ( counter2 == 0 )
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
  if ( counter2 == 0 )
  {
    fprintf(fp, "from scipy.io.array_import import *\n");
    fprintf(fp, "\n");
    fprintf(fp, "t = read_array(\"%s\")\n", filename_t.c_str());
    fprintf(fp, "u = read_array(\"%s\")\n", filename_u.c_str());
    fprintf(fp, "k = read_array(\"%s\")\n", filename_k.c_str());
    fprintf(fp, "r = read_array(\"%s\")\n", filename_r.c_str());
  }

  // Save time
  fprintf(fp_t, "%.15e\n", sample.t());

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
  counter2++;
  
  // Close files
  fclose(fp_t);
  fclose(fp_u);
  fclose(fp_k);
  fclose(fp_r);

  // Close file
  fclose(fp);

}
//-----------------------------------------------------------------------------
