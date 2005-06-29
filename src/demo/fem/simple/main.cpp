// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Thanks to David Heintz for the reference matrices.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  class hej { int a; };

  // Load reference mesh and matrices
  Mesh mesh("tetrahedron.xml.gz");

  // Create stiffness and mass matrices
  StiffnessMatrix A(mesh);
  MassMatrix M(mesh);

  // Create reference matrices
  Matrix A0(4,4);
  Matrix M0(4,4);

  A0(0,0) =  1.0/2.0;   A0(0,1) = -1.0/6.0;   A0(0,2) = -1.0/6.0;   A0(0,3) = -1.0/6.0;
  A0(1,0) = -1.0/6.0;   A0(1,1) =  1.0/6.0;   A0(1,2) =  0.0;       A0(1,3) =  0.0;
  A0(2,0) = -1.0/6.0;   A0(2,1) =  0.0;       A0(2,2) =  1.0/6.0;   A0(2,3) =  0.0;
  A0(3,0) = -1.0/6.0;   A0(3,1) =  0.0;       A0(3,2) =  0.0;       A0(3,3) =  1.0/6.0;

  M0(0,0) =  1.0/60.0;  M0(0,1) =  1.0/120.0; M0(0,2) =  1.0/120.0; M0(0,3) =  1.0/120.0;
  M0(1,0) =  1.0/120.0; M0(1,1) =  1.0/60.0;  M0(1,2) =  1.0/120.0; M0(1,3) =  1.0/120.0;
  M0(2,0) =  1.0/120.0; M0(2,1) =  1.0/120.0; M0(2,2) =  1.0/60.0;  M0(2,3) =  1.0/120.0;
  M0(3,0) =  1.0/120.0; M0(3,1) =  1.0/120.0; M0(3,2) =  1.0/120.0; M0(3,3) =  1.0/60.0;

  // Disp matrices
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
