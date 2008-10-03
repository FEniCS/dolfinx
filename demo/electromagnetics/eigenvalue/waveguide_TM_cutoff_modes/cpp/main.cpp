// Copyright (C) 2008 Evan Lezar (evanlezar@gmail.com).
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-22
// Last changed: 2008-10-02
//
// This demo demonstrates the calculation of a TM (Transverse Magnetic)
// cutoff wavenumber of a rectangular waveguide with dimensions 1x0.5m.
//
// For more information regarding waveguides see
//
//   http://www.ee.bilkent.edu.tr/~microwave/programs/magnetic/rect/info.htm
//
// See the pdf in the parent folder and the following reference
//
// The Finite Element in Electromagnetics (2nd Ed)
// Jianming Jin [7.2.1 - 7.2.2]

#include <dolfin.h>
#include "Forms.h"

using namespace dolfin;

#ifdef HAS_PETSC

int main()
{
  float width = 1.0;
  float height = 0.5;
  
  Rectangle mesh(0, width, 0, height, 2, 1);
  
  // Define the forms
  Forms_0 a;
  Forms_1 L;
  
  // Assemble the system matrices stiffness (S) and mass matrices (T)
  PETScMatrix S;
  PETScMatrix T;
  Assembler assembler(mesh);
  
  assembler.assemble(S, a);
  assembler.assemble(T, L);
  
  // Solve the eigen system
  SLEPcEigenSolver esolver;
  
  esolver.set("eigenvalue spectrum", "smallest real");
  esolver.set("eigenvalue tolerance", 10e-7);
  esolver.set("eigenvalue iterations", 10);
  esolver.solve(S, T);
  
  int dominant_mode_index = -1;
  real lr, lc;
  for ( int i = 0; i < (int)S.size(1); i++ )
  {
    esolver.getEigenvalue(lr, lc, i);
    //printf("Eigenvalue %d : %f + i%f\n", i, lr, lc);
    //ensure that the real part is large enough and that the complex part is zero
    if ((lr > 1) && (lc == 0))
    {
      message("Dominant mode found\n");
      dominant_mode_index = i;
      break;
    }
  }
  
  if (dominant_mode_index >= 0)
  {
    message("Dominant mode found: %f\n", lr);
    if ( lc != 0 )
      warning("Cutoff mode is complex (%f + i%f)\n", lr, lc);
  }
  else
    error("Cutoff mode not found.");
  
  return 0;
}

#else

int main()
{
  message("Sorry, this demo is not available when DOLFIN is compiled without PETSc.");
  return 0;
}

#endif
