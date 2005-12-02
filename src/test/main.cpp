// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005-11-29
//
// This file is used for testing out new features implemented
// in the library, which means that the contents of this file
// might be (should be) constantly changing. Anything can be
// thrown into this file at any time and may also be removed
// at any time. Use this for temporary test programs that are
// not suitable to be implemented as demos in src/demo.

#include <dolfin.h>
#include "L2Norm.h"

using namespace dolfin;

class MyPreconditioner : public Preconditioner
{
public:

  MyPreconditioner(const Vector& m) : m(m) {}

  void solve(Vector& x, const Vector& b)
  {
    dolfin_info("Calling preconditioner");
    for (unsigned int i = 0; i < x.size(); i++)
    {
      x(i) = b(i) / m(i);
    }
  }

private:

  const Vector& m;

};

void testPreconditioner()
{
  UnitSquare mesh(2, 2);
  MassMatrix M(mesh);
  Vector x;
  Vector b(M.size(0));
  b = 1.0;
  Vector m;
  FEM::lump(M, m);

 
  GMRES solver;
  MyPreconditioner pc(m);
  solver.setPreconditioner(pc);
  solver.solve(M, x, b);
  
  x.disp();
}

void testMatrixOutput()
{
  UnitSquare mesh(2, 2);
  MassMatrix M(mesh);

  File file("matrix.m");
  file << M;
}

class MyFunction0 : public Function
{
public:

  real eval(const Point& p, unsigned int i)
  {
    return p.x;
  }

};

class MyFunction1 : public Function
{
public:

  real eval(const Point& p, unsigned int i)
  {
    return p.y;
  }

};

void testFunctional()
{
  dolfin_info("Computing L2 norm of f(x, y) = x - y on the unit square.");

  UnitSquare mesh(16, 16);
  MyFunction0 f;
  MyFunction1 g;
  L2Norm::LinearForm L(f, g);
  Vector b;
  FEM::assemble(L, b, mesh);
  real norm = sqrt(b.sum());

  dolfin_info("Result:   %.15f", norm);
  dolfin_info("Analytic: 0.408248290463863");
}

void testOutput()
{
  Mesh mesh("dolfin-1.xml.gz");
  File file("mesh.m");
  file << mesh;
}

int main(int argc, char* argv[])
{
  dolfin_info("Testing DOLFIN...");

  //testOutput();

  return 0;
}
