// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2006-03-01
//
// This file is used for testing out new features implemented in the
// library, which means that the contents of this file is constantly
// changing. Anything can be thrown into this file at any time. Use
// this for simple tests that are not suitable to be implemented as
// demos in src/demo.

#include <dolfin.h>
#include <dolfin/Poisson2D.h>
#include "L2Norm.h"

using namespace dolfin;

void testPreconditioner()
{
  dolfin_info("--- Testing preconditioner ---");

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

void testOutputVector()
{
  dolfin_info("--- Testing output of vector ---");

  Vector x(10);
  for (unsigned int i = 0; i < 10; i++)
    x(i) = 1.0 / static_cast<real>(i + 1);

  File mfile("vector.m");
  mfile << x;

  File xmlfile("vector.xml");
  xmlfile << x;
}

void testOutputMatrix()
{
  dolfin_info("--- Testing output of matrix ---");

  UnitSquare mesh(2, 2);
  MassMatrix M(mesh);

  File mfile("matrix.m");
  mfile << M;

  File xmlfile("matrix.xml");
  xmlfile << M;
}

void testOutputMesh()
{
  dolfin_info("--- Testing output of mesh ---");

  UnitCube mesh(16, 16, 16);
  File file("mesh.m");
  file << mesh;
}

void testOutputFiniteElementSpec()
{
  dolfin_info("--- Testing output of finite element specification ---");

  FiniteElementSpec spec("Lagrange", "triangle", 1, 1);
  File file("finiteelement.xml");
  file << spec;
}

void testOutputFunction()
{
  dolfin_info("--- Testing output of function ---");

  UnitSquare mesh(4, 4);
  Vector x(mesh.numVertices());
  P1tri element;

  Function f(x, mesh, element);

  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    Point & p = v->coord();
    x(v->id()) = sin(2.0*p.x) * cos(3.0*p.y);
  }

  File file("function.xml");
  file << f;
}

void testOutputMultiple()
{
  dolfin_info("--- Testing input of multiple XML objects ---");

  Vector x(3);
  Matrix A(3, 3);
  A(1, 1) = 1.0;

  File file("multiple.xml");
  file << x;
  file << A;
  file << A;
}

void testInputFunction()
{
  dolfin_info("--- Testing input of function ---");

  Function f;
  File file("function.xml");
  file >> f;

  for (VertexIterator v(f.mesh()); !v.end(); ++v)
    cout << v->coord() << ": f = " << f(*v) << endl;
}

void testFunctional()
{
  dolfin_info("--- Testing functional ---");

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

void testRandom()
{
  dolfin_info("--- Testing random ---");

  cout << "Random numbers: ";
  cout << dolfin::rand() << " ";
  cout << dolfin::rand() << " ";
  cout << dolfin::rand() << endl;
}

void testProgress()
{
  dolfin_info("--- Testing progress ---");

  Progress p("Testing progress bar", 500);
  for (unsigned int i = 0; i < 500; i++)
    p++;
}

void testParameters()
{
  dolfin_info("--- Testing parameters ---");

  class Foo : public Parametrized {};
  class Bar : public Parametrized {};

  Foo foo;
  Bar bar;

  bar.set("parent", foo);

  foo.set("tolerance", 0.00007);
  foo.set("solution file name", "foo.tmp");

  bar.set("solution file name", "bar.tmp");
  bar.set("my parameter", "bar");

  cout << "global parameters:" << endl;
  cout << "  " << get("method")    << endl;
  cout << "  " << get("tolerance") << endl;
  cout << "  " << get("solution file name") << endl;

  cout << "parameters for foo: "       << endl;
  cout << "  " << foo.get("method")    << endl;
  cout << "  " << foo.get("tolerance") << endl;
  cout << "  " << foo.get("solution file name") << endl;

  cout << "parameters for bar: "          << endl;
  cout << "  " << bar.get("method")       << endl;
  cout << "  " << bar.get("tolerance")    << endl;
  cout << "  " << bar.get("solution file name")    << endl;
  cout << "  " << bar.get("my parameter") << endl;
}

void testMakeElement()
{
  dolfin_info("--- Testing creation of element from spec ---");
  
  FiniteElement* P1 = FiniteElement::makeElement("Lagrange", "triangle", 1);
  FiniteElement* P2 = FiniteElement::makeElement("Lagrange", "triangle", 2);

  if ( P1 ) delete P1;
  if ( P2 ) delete P2;
}

int main(int argc, char* argv[])
{
  testPreconditioner();
  testOutputVector();
  testOutputMatrix();
  testOutputMesh();
  testOutputFiniteElementSpec();
  testOutputFunction();
  testOutputMultiple();
  testInputFunction();
  testFunctional();
  testRandom();
  testProgress();
  testParameters();
  testMakeElement();

  return 0;
}
