// Copyright (C) 2008 Evan Lezar
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2008, 2015
//
// First added:  2008-08-22
// Last changed: 2015-06-15
//
// This demo demonstrates the calculation of a TM (Transverse Magnetic)
// cutoff wavenumber of a rectangular waveguide with dimensions 1x0.5m.
//
// For more information regarding waveguides, see
//
//   http://www.ee.bilkent.edu.tr/~microwave/programs/magnetic/rect/info.htm
//
// See the pdf in the parent folder and the following reference:
//
//   The Finite Element in Electromagnetics (2nd Ed)
//   Jianming Jin [7.2.1 - 7.2.2]

#include <dolfin.h>
#include "Forms.h"

using namespace dolfin;

#if defined(HAS_PETSC) && defined(HAS_SLEPC)

int main()
{
  // Create mesh
  double width = 1.0;
  double height = 0.5;
  RectangleMesh mesh(Point(0.0, 0.0), Point(width, height), 4, 2);

  // Define the forms - gererates an generalized eigenproblem of the form
  // [S]{h} = k_o^2[T]{h}
  // with the eigenvalues k_o^2 representing the square of the cutoff wavenumber
  // and the corresponding right-eigenvector giving the coefficients of the
  // discrete system used to obtain the approximate field anywhere in the domain
  Forms::FunctionSpace V(mesh);
  Forms::Form_a s(V, V);
  Forms::Form_L t(V, V);

  // Assemble the system matrices stiffness (S) and mass matrices (T)
  auto S = std::make_shared<PETScMatrix>();
  auto T = std::make_shared<PETScMatrix>();
  assemble(*S, s);
  assemble(*T, t);

  // Solve the eigen system
  SLEPcEigenSolver esolver(S, T);
  esolver.parameters["spectrum"] = "smallest real";
  esolver.parameters["solver"] = "lapack";
  esolver.solve();

  // The result should have real eigenvalues but due to rounding
  // errors, some of the resultant eigenvalues may be small complex
  // values.  only consider the real part

  // Now, the system contains a number of zero eigenvalues (near zero
  // due to rounding) which are eigenvalues corresponding to the
  // null-space of the curl operator and are a mathematical construct
  // and do not represent physically realizable modes.  These are
  // called spurious modes.  So, we need to identify the smallest,
  // non-zero eigenvalue of the system - which corresponds with cutoff
  // wavenumber of the the dominant cutoff mode.
  double cutoff = -1.0;
  double lr, lc;
  for (std::size_t i = 0; i < S->size(1); i++)
  {
    esolver.get_eigenvalue(lr, lc, i);
    if (lr > 1 && lc == 0)
    {
      cutoff = sqrt(lr);
      break;
    }
  }

  if (cutoff == -1.0)
    info("Unable to find dominant mode.");
  else
    info("Cutoff frequency = %g", cutoff);

  return 0;
}

#else

int main()
{
  info("Sorry, this demo is only available when DOLFIN is compiled with PETSc and SLEPc.");
  return 0;
}

#endif
