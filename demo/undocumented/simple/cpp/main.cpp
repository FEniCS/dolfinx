// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2007, 2010.
//
// Thanks to David Heintz for the reference matrices.
//
// This demo program dmonstrates how to create simple finite
// element matrices like the stiffness matrix and mass matrix.
// For general forms and matrices, forms must be defined and
// compiled with FFC.

#include <dolfin.h>
#include "MassMatrix3D.h"
#include "StiffnessMatrix3D.h"

using namespace dolfin;

int main()
{
  not_working_in_parallel("This demo");

  // Load reference mesh (just a simple tetrahedron)
  Mesh mesh("../tetrahedron.xml.gz");

  // Create stiffness and mass matrices
  uBLASDenseMatrix A, M;
  StiffnessMatrix3D::FunctionSpace V0(mesh);
  StiffnessMatrix3D::BilinearForm a0(V0, V0);
  assemble(A, a0);
  MassMatrix3D::FunctionSpace V1(mesh);
  MassMatrix3D::BilinearForm a1(V1, V1);
  assemble(M, a1);

  // Create reference matrices
  double A0_array[4][4];
  double M0_array[4][4];

  A0_array[0][0] = 1.0/2.0;   A0_array[0][1] =-1.0/6.0;   A0_array[0][2] =-1.0/6.0;   A0_array[0][3] =-1.0/6.0;
  A0_array[1][0] =-1.0/6.0;   A0_array[1][1] = 1.0/6.0;   A0_array[1][2] = 0.0;       A0_array[1][3] = 0.0;
  A0_array[2][0] =-1.0/6.0;   A0_array[2][1] = 0.0;       A0_array[2][2] = 1.0/6.0;   A0_array[2][3] = 0.0;
  A0_array[3][0] =-1.0/6.0;   A0_array[3][1] = 0.0;       A0_array[3][2] = 0.0;       A0_array[3][3] = 1.0/6.0;

  M0_array[0][0] = 1.0/60.0;  M0_array[0][1] = 1.0/120.0; M0_array[0][2] = 1.0/120.0; M0_array[0][3] = 1.0/120.0;
  M0_array[1][0] = 1.0/120.0; M0_array[1][1] = 1.0/60.0;  M0_array[1][2] = 1.0/120.0; M0_array[1][3] = 1.0/120.0;
  M0_array[2][0] = 1.0/120.0; M0_array[2][1] = 1.0/120.0; M0_array[2][2] = 1.0/60.0;  M0_array[2][3] = 1.0/120.0;
  M0_array[3][0] = 1.0/120.0; M0_array[3][1] = 1.0/120.0; M0_array[3][2] = 1.0/120.0; M0_array[3][3] = 1.0/60.0;

  unsigned int position[4] = {0, 1, 2, 3};

  uBLASDenseMatrix A0(4, 4);
  uBLASDenseMatrix M0(4, 4);
  A0.set(*A0_array, 4, position, 4, position);
  M0.set(*M0_array, 4, position, 4, position);

  A0.apply("insert");
  M0.apply("insert");

  // Display matrices
  cout << endl;
  cout << "Assembled stiffness matrix:" << endl;
  info(A, true);
  cout << endl;

  cout << "Reference stiffness matrix:" << endl;
  info(A0, true);
  cout << endl;

  cout << "Assembled mass matrix:" << endl;
  info(M, true);
  cout << endl;

  cout << "Reference mass matrix:" << endl;
  info(M0, true);
  cout << endl;

  return 0;
}
