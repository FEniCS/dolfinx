// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005.
// Modified by Rolv E. Bredesen 2008.

// First added:  2003-05-06
// Last changed: 2008-04-08
// 

#include <dolfin/log/dolfin_log.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/function/Function.h>
#include <dolfin/ode/Sample.h>
#include "PythonFile.h"

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

  #ifdef HAS_GMP
  warning("PythonFile: Precision lost. Values will be saved with double precision");  
  #endif
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
    fprintf(fp, "from numpy import fromfile\n");
    fprintf(fp, "\n");
    fprintf(fp, "t = fromfile(\"%s\", sep=\" \")\n", filename_t.c_str());
    fprintf(fp, "u = fromfile(\"%s\", sep=\" \")\n", filename_u.c_str());
    fprintf(fp, "k = fromfile(\"%s\", sep=\" \")\n", filename_k.c_str());
    fprintf(fp, "r = fromfile(\"%s\", sep=\" \")\n", filename_r.c_str());
    fprintf(fp, "\n");
    fprintf(fp, "u.shape = len(u)//%d, %d\n", sample.size(), sample.size());
    fprintf(fp, "k.shape = len(k)//%d, %d\n", sample.size(), sample.size());
    fprintf(fp, "r.shape = len(r)//%d, %d\n", sample.size(), sample.size());
    fprintf(fp, "\n");
  }

  // Save time
  fprintf(fp_t, "%.15e\n", to_double(sample.t()));

  // Save solution
  for (unsigned int i = 0; i < sample.size(); i++)
  {
    fprintf(fp_u, "%.15e ", to_double(sample.u(i)));  
  }
  fprintf(fp_u, "\n");

  // Save time steps
  for (unsigned int i = 0; i < sample.size(); i++)
  {
    fprintf(fp_k, "%.15e ", to_double(sample.k(i)));  
  }
  fprintf(fp_k, "\n");

  // Save residuals
  for (unsigned int i = 0; i < sample.size(); i++)
  {
    fprintf(fp_r, "%.15e ", to_double(sample.r(i)));  
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
