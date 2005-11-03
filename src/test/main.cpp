// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.

// This file is used for testing out new features implemented
// in the library, which means that the contents of this file
// might be (should be) constantly changing. Anything can be
// thrown into this file at any time and may also be removed
// at any time. Use this for temporary test programs that are
// not suitable to be implemented as demos in src/demo.

#include <dolfin.h>

using namespace dolfin;

class MyPreconditioner : public Preconditioner
{
public:

  void solve(Vector& x, const Vector& b)
  {
    x = b;
    dolfin_info("Calling preconditioner");
  }

};

int main(int argc, char* argv[])
{
  dolfin_info("Testing DOLFIN...");

  UnitSquare mesh(2, 2);
  MassMatrix A(mesh);
  Vector x;
  Vector b(A.size(0));
  b = 1.0;

  GMRES solver;
  MyPreconditioner pc;
  solver.setPreconditioner(pc);
  solver.solve(A, x, b);
  
  x.disp();

  return 0;
}
