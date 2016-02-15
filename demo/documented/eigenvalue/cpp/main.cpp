// Copyright (C) 2007-2010 Kristian B. Oelgaard and Garth N. Wells
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
// Modified by Anders Logg, 2008.
// Modified by Marie E. Rognes, 2010.
//
// First added:  2007-03-08
// Last changed: 2012-07-05
//
// This simple program illustrates the use of the SLEPc eigenvalue solver.

#include <dolfin.h>
#include "StiffnessMatrix.h"

using namespace dolfin;

int main()
{
  #ifdef HAS_SLEPC

  // Create mesh
  auto mesh = std::make_shared<Mesh>("../box_with_dent.xml.gz");

  // Build stiffness matrix
  auto A = std::make_shared<PETScMatrix>();
  auto V = std::make_shared<StiffnessMatrix::FunctionSpace>(mesh);
  StiffnessMatrix::BilinearForm a(V, V);
  assemble(*A, a);

  // Create eigensolver
  SLEPcEigenSolver esolver(A);

  // Compute all eigenvalues of A x = \lambda x
  esolver.solve();

  // Extract largest (first, n =0) eigenpair
  double r, c;
  PETScVector rx, cx;
  esolver.get_eigenpair(r, c, rx, cx, 0);

  cout << "Largest eigenvalue: " << r << endl;

  // Initialize function with eigenvector
  Function u(V);
  *u.vector() = rx;

  // Plot eigenfunction
  plot(u);
  interactive();

  #else

  dolfin::cout << "SLEPc must be installed to run this demo." << dolfin::endl;

  #endif

  return 0;
}
