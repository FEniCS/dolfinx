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
  // Specify the waveguide width and height in metres
  float width = 1.0;
  float height = 0.5;
  
  // Create the mesh using a Rectangle
  Rectangle mesh(0, width, 0, height, 2, 1);

  // Define the forms - gererates an generalized eigenproblem of the form 
  // [S]{h} = k_o^2[T]{h}
  // with the eigenvalues k_o^2 representing the square of the cutoff wavenumber 
  // and the corresponding right-eigenvector giving the coefficients of the 
  // discrete system used to obtain the approximate field anywhere in the domain   
  Forms_0 s;
  Forms_1 t;

  // Assemble the system matrices stiffness (S) and mass matrices (T)
  PETScMatrix S;
  PETScMatrix T;
  Assembler assembler(mesh);

  assembler.assemble(S, s);
  assembler.assemble(T, t);

  // Solve the eigen system
  SLEPcEigenSolver esolver;
  esolver.set("eigenvalue spectrum", "smallest real");
  esolver.set("eigenvalue solver", "lapack");
  esolver.solve(S, T);
  
  // The result should have real eigenvalues but due to rounding errors, some of 
  // the resultant eigenvalues may be small complex values. 
  // only consider the real part

  // Now, the system contains a number of zero eigenvalues (near zero due to 
  // rounding) which are eigenvalues corresponding to the null-space of the curl 
  // operator and are a mathematical construct and do not represent physically 
  // realizable modes.  These are called spurious modes.  
  // So, we need to identify the smallest, non-zero eigenvalue of the system - 
  // which corresponds with cutoff wavenumber of the the dominant cutoff mode.
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
    message("Dominant mode found.  Cuttoff Squared: %f\n", lr);
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
