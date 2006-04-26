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


  MyPreconditioner pc(m); 
  GMRES solver(pc);
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

  UnitCube mesh(8, 8, 8);
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

void testMeshRefinement()
{
  dolfin_info("--- Testing mesh refinement ---");
  
  UnitSquare mesh(2, 2);

  File file0("mesh0.pvd");
  File file1("mesh1.pvd");
  File file2("mesh2.pvd");

  //file0 << mesh;

  // Mark one cell for refinement
  //mesh.cell(0).mark();
  //mesh.refine();
  //file1 << mesh;

  // Mark another cell in parent for refinement
  //Mesh mesh1(mesh);
  //parent.cell(7).mark();
  // parent.refine();
  //file2 << mesh;
}

void testDenseMatrix()
{
  dolfin_info("--- Testing dense matrices ---");
  
  const int kk   = 100000;
  const int size = 6;

  real time;

  cout << "Assemble a "<< size << "x" << size << " matrix " << kk << " times." << endl;

  dolfin_begin("Setting values");
  tic();
  boost::numeric::ublas::matrix<real> A(size, size); 
  for(int k=0; k < kk; ++k) 
   for(int i=0; i < size; ++i)
      for(int j=0; j < size; ++j)
        A(i,j) = 1.0;

  time = toc();
  cout << "Plain ublas time to set values = " << time << endl;

  tic();
  DenseMatrix B(size, size); 
  for(int k=0; k < kk; ++k) 
   for(int i=0; i < size; ++i)
      for(int j=0; j < size; ++j)
        B(i,j) = 1.0;
  time = toc();
  cout << "Dense matrix time to set values = " << time << endl;

  tic();
  NewMatrix<DenseMatrix> D(size, size); 
  for(int k=0; k < kk; ++k) 
   for(int i=0; i < size; ++i)
      for(int j=0; j < size; ++j)
        D(i,j) = 1.7;
  time = toc();
  cout << "Template NewMatrix time to set values = " << time << endl;
   
  dolfin_end();



  dolfin_begin("Retrieving values");
  real temp;

  tic();
  for(int k=0; k < kk; ++k) 
   for(int i=0; i < size; ++i)
      for(int j=0; j < size; ++j)
        temp = A(i,j);
  time = toc();
  cout << "Plain ublas time to retrieve values = " << time << endl;

  tic();
  for(int k=0; k < kk; ++k) 
   for(int i=0; i < size; ++i)
      for(int j=0; j < size; ++j)
        temp = B(i,j);
  time = toc();
  cout << "Dense matrix time to retrieve values = " << time << endl;


  tic();
  for(int k=0; k < kk; ++k) 
   for(int i=0; i < size; ++i)
      for(int j=0; j < size; ++j)
        temp = D(i,j);
  time = toc();
  cout << "Template NewMatrix time to retrieve values = " << time << endl;


  // Assemble stiffness matrix into a dense and a sparse matrix
  UnitSquare mesh(1, 1);
  
  Function f = 1.0;
  Poisson2D::BilinearForm a;
  Poisson2D::LinearForm L(f);
  
  class MyBC : public BoundaryCondition
  {
    void eval(BoundaryValue& value, const Point& p, unsigned int i)
    {
	      value = 1.0;
    }
  };

  MyBC bc;

  // Assemble dense vectors and matrices
  DenseMatrix K;
  DenseVector b;
  FEM::assemble(a, K, mesh); 
  FEM::assemble(L, b, mesh); 
  FEM::assemble(a, L, K, b, mesh); 
//  FEM::assemble(a, L, K, b, mesh, bc); 
  K.disp();
  b.disp();  
    
  // Assemble sparse vectors and matrices
  Matrix M;
  Vector g;
  FEM::assemble(a, M, mesh); 
  FEM::assemble(L, g, mesh); 
  FEM::assemble(a, L, M, g, mesh); 
  FEM::assemble(a, L, M, g, mesh, bc); 
  M.disp();
  g.disp();  
  
  // Assemble mixture of sparse and dense vectors and matrices
  FEM::assemble(a, L, K, g, mesh); 
  FEM::assemble(a, L, M, b, mesh); 

}


int main(int argc, char* argv[])
{
/*
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
  testMeshRefinement();
*/
  testDenseMatrix();

  return 0;
}
