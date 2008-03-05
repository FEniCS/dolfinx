// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
//
// Thanks to David Heintz for the reference matrices.
//
// This demo program dmonstrates how to create simple finite
// element matrices like the stiffness matrix and mass matrix.
// For general forms and matrices, forms must be defined and
// compiled with FFC.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Load reference mesh (just a simple tetrahedron)
  Mesh mesh("../tetrahedron.xml.gz");

  // Create stiffness and mass matrices
  StiffnessMatrix A(mesh);
  MassMatrix M(mesh);

  // Create reference matrices
  real A0_array[4][4];
  real M0_array[4][4];

  A0_array[0][0] = 1.0/2.0;   A0_array[0][1] =-1.0/6.0;   A0_array[0][2] =-1.0/6.0;   A0_array[0][3] =-1.0/6.0;
  A0_array[1][0] =-1.0/6.0;   A0_array[1][1] = 1.0/6.0;   A0_array[1][2] = 0.0;       A0_array[1][3] = 0.0;
  A0_array[2][0] =-1.0/6.0;   A0_array[2][1] = 0.0;       A0_array[2][2] = 1.0/6.0;   A0_array[2][3] = 0.0;
  A0_array[3][0] =-1.0/6.0;   A0_array[3][1] = 0.0;       A0_array[3][2] = 0.0;       A0_array[3][3] = 1.0/6.0;

  M0_array[0][0] = 1.0/60.0;  M0_array[0][1] = 1.0/120.0; M0_array[0][2] = 1.0/120.0; M0_array[0][3] = 1.0/120.0;
  M0_array[1][0] = 1.0/120.0; M0_array[1][1] = 1.0/60.0;  M0_array[1][2] = 1.0/120.0; M0_array[1][3] = 1.0/120.0;
  M0_array[2][0] = 1.0/120.0; M0_array[2][1] = 1.0/120.0; M0_array[2][2] = 1.0/60.0;  M0_array[2][3] = 1.0/120.0;
  M0_array[3][0] = 1.0/120.0; M0_array[3][1] = 1.0/120.0; M0_array[3][2] = 1.0/120.0; M0_array[3][3] = 1.0/60.0;

  unsigned int position[4] = {0, 1, 2, 3};

  Matrix A0(4,4);
  Matrix M0(4,4);
  A0.set(*A0_array, 4, position, 4, position);
  M0.set(*M0_array, 4, position, 4, position);

  A0.apply(); 
  M0.apply(); 

  // Display matrices
  cout << endl;
  cout << "Assembled stiffness matrix:" << endl;
  A.disp();
  cout << endl;

  cout << "Reference stiffness matrix:" << endl;
  A0.disp();
  cout << endl;

  cout << "Assembled mass matrix:" << endl;
  M.disp();
  cout << endl;

  cout << "Reference mass matrix:" << endl;
  M0.disp();
  cout << endl;

  return 0;
}
